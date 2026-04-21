"""
Wordle policy using a pretrained Hugging Face causal LM plus a trainable linear head
over a fixed 5-letter vocabulary. Supports raw-text tokenization for GPT-2-style models
and chat-template tokenization for instruction-tuned models such as Gemma.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .wordle_env import WordVocabulary, WordleState, MOCK_WORDLE_TARGETS
    from .wordle_hints import build_constraint_summary
except ImportError:
    from wordle_env import WordVocabulary, WordleState, MOCK_WORDLE_TARGETS
    from wordle_hints import build_constraint_summary


def _words_with_priority(
    full_vocab: WordVocabulary,
    max_vocab_size: Optional[int],
    priority: List[str],
) -> List[str]:
    """
    Build a fixed action list: take valid priority words first, then scan the full
    lexicographic list until `max_vocab_size` (so truncated vocabs still cover mock targets).
    """
    seen: set = set()
    out: List[str] = []
    for w in priority:
        u = w.upper()
        if u in full_vocab.word_to_idx and u not in seen:
            out.append(u)
            seen.add(u)
    for w in full_vocab.words:
        if max_vocab_size is not None and len(out) >= max_vocab_size:
            break
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def _build_wordle_prompt(state: WordleState, richer_prompt: bool = True) -> str:
    """Turn game state into a short natural-language prompt for the LM."""
    lines = [
        "You are playing Wordle. Guess a valid 5-letter English word.",
        f"Turn {state.turn_number + 1} of 6.",
    ]
    if state.previous_guesses:
        lines.append("Previous guesses and feedback:")
        for g, fb in zip(state.previous_guesses, state.feedback_history):
            lines.append(f"- {g}: {fb}")
        if richer_prompt:
            hint = build_constraint_summary(state.previous_guesses, state.feedback_history)
            if hint:
                lines.append(hint)
    else:
        lines.append("No guesses yet. Choose a strong opening word.")
    lines.append("Reply with exactly one 5-letter uppercase word as your next guess.")
    return "\n".join(lines)


def _default_use_chat_template(model_name: str) -> bool:
    lowered = model_name.lower()
    return any(tag in lowered for tag in ("gemma", "instruct", "-it", "chat"))


def _default_lora_targets(model_name: str) -> List[str]:
    lowered = model_name.lower()
    if "gpt2" in lowered:
        return ["c_attn", "c_proj"]
    if "gemma" in lowered:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    raise ValueError(
        "No default LoRA target modules are defined for "
        f"{model_name!r}. Pass `lora_target_modules=` explicitly."
    )


def _version_tuple(version: str) -> Tuple[int, ...]:
    parts = []
    for piece in version.split("."):
        match = re.match(r"(\d+)", piece)
        if not match:
            break
        parts.append(int(match.group(1)))
    return tuple(parts)


def _require_jinja2_for_chat_templates() -> None:
    try:
        import jinja2
    except ImportError as exc:
        raise ImportError(
            "Chat-template tokenization requires `jinja2>=3.1.0`. "
            "Install or upgrade it with: `python -m pip install -U 'jinja2>=3.1.0'`."
        ) from exc

    installed = getattr(jinja2, "__version__", "0")
    if _version_tuple(installed) < (3, 1, 0):
        raise ImportError(
            "Chat-template tokenization requires `jinja2>=3.1.0`, "
            f"but found {installed}. Upgrade it with: "
            "`python -m pip install -U 'jinja2>=3.1.0'`."
        )


def _filter_model_inputs(enc: Dict[str, torch.Tensor], allowed_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    allowed = set(allowed_keys)
    return {k: v for k, v in enc.items() if k in allowed}


def _extract_last_hidden_state(model_output: Any) -> torch.Tensor:
    last_hidden_state = getattr(model_output, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state

    hidden_states = getattr(model_output, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]

    raise RuntimeError("Model output did not include hidden states.")


def _last_non_padding_index(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
    masked_positions = positions * attention_mask.long()
    return masked_positions.max(dim=1).values


class WordleGPT2Policy(nn.Module):
    """
    Hugging Face causal LM encoder + trainable logits head over a vocabulary subset.
    Optional LoRA on model-specific projection layers (`use_lora=True`); otherwise the
    backbone is frozen and only the head is trained.

    `state_embedding` arguments in `get_action` / `format_action_xml` are ignored; the
    model conditions on `WordleState` via the text prompt (same information as the env).
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        use_prime_targets: bool = False,
        max_vocab_size: Optional[int] = 512,
        max_prompt_length: int = 256,
        include_mock_targets_in_vocab: bool = True,
        extra_priority_words: Optional[List[str]] = None,
        richer_prompt: bool = True,
        use_chat_template: Optional[bool] = None,
        chat_generation_prompt: bool = True,
        use_lora: bool = False,
        action_granularity: str = "char",
        lora_r: int = 4,
        lora_alpha: float = 16.0,
        lora_target_modules: Optional[List[str]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "wordle_gpt2_policy requires `transformers`. Install with: pip install transformers"
            )
        if use_lora and not PEFT_AVAILABLE:
            raise ImportError("LoRA requires `peft`. Install with: pip install peft")

        full_vocab = WordVocabulary(use_prime_targets=use_prime_targets)
        priority: List[str] = []
        if include_mock_targets_in_vocab:
            priority.extend(MOCK_WORDLE_TARGETS)
        if extra_priority_words:
            priority.extend(extra_priority_words)
        # Snapshot priority so expand_vocab() can reproduce the same prefix
        # ordering and preserve the append-only invariant when N grows.
        self._priority: List[str] = list(priority)
        words = _words_with_priority(full_vocab, max_vocab_size, priority)

        self.words: List[str] = words
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for i, w in enumerate(self.words)}
        self.action_dim = len(self.words)
        # Alias for code that expects `policy.vocab`
        self.vocab = full_vocab

        self.richer_prompt = richer_prompt
        self.use_chat_template = (
            _default_use_chat_template(model_name)
            if use_chat_template is None
            else use_chat_template
        )
        if self.use_chat_template:
            _require_jinja2_for_chat_templates()
        self.chat_generation_prompt = chat_generation_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **(tokenizer_kwargs or {}))
        self.lm = AutoModelForCausalLM.from_pretrained(model_name, **(model_kwargs or {}))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._lm_forward_keys = tuple(inspect.signature(self.lm.forward).parameters.keys())

        self._lm_trainable = False
        if use_lora:
            target_modules = (
                list(lora_target_modules)
                if lora_target_modules is not None
                else _default_lora_targets(model_name)
            )
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=int(lora_alpha),
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, peft_config)
            self._lm_trainable = True
        else:
            for p in self.lm.parameters():
                p.requires_grad = False

        hidden = int(getattr(self.lm.config, "hidden_size", getattr(self.lm.config, "dim", 768)))
        self.head = nn.Linear(hidden, self.action_dim)
        self.max_prompt_length = max_prompt_length
        self._model_name = model_name
        self.action_granularity = action_granularity.lower().strip()
        if self.action_granularity not in {"word", "char"}:
            raise ValueError(
                f"action_granularity must be 'word' or 'char', got {action_granularity!r}."
            )

        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)
        self.char_head = nn.Linear(hidden, 26)
        nn.init.orthogonal_(self.char_head.weight, gain=0.01)
        nn.init.zeros_(self.char_head.bias)

    @staticmethod
    def _letter_to_idx(letter: str) -> int:
        return ord(letter) - ord("A")

    @staticmethod
    def _idx_to_letter(idx: int) -> str:
        return chr(ord("A") + idx)

    def _feedback_constraints(self, state: WordleState) -> Dict[str, Any]:
        fixed: Dict[int, str] = {}
        blocked_pos: Dict[int, Set[str]] = {i: set() for i in range(5)}
        required: Set[str] = set()
        seen_non_gray: Set[str] = set()
        gray_only: Set[str] = set()
        for guess, feedback in zip(state.previous_guesses, state.feedback_history):
            parts = feedback.split()
            if len(parts) != 5:
                continue
            for i, part in enumerate(parts):
                if ":" not in part:
                    continue
                letter, color = part.split(":", 1)
                letter = letter.upper()
                color = color.upper()
                if len(letter) != 1 or not letter.isalpha():
                    continue
                if color == "GREEN":
                    fixed[i] = letter
                    required.add(letter)
                    seen_non_gray.add(letter)
                elif color == "YELLOW":
                    blocked_pos[i].add(letter)
                    required.add(letter)
                    seen_non_gray.add(letter)
                elif color == "GRAY":
                    gray_only.add(letter)
        banned_global = {c for c in gray_only if c not in seen_non_gray}
        return {
            "fixed": fixed,
            "blocked_pos": blocked_pos,
            "required": required,
            "banned_global": banned_global,
        }

    def _char_logits_for_partial(self, state: WordleState, partial: str) -> torch.Tensor:
        """Get next-letter logits conditioned on state and current partial guess."""
        text = _build_wordle_prompt(state, richer_prompt=self.richer_prompt)
        text += (
            f"\nCurrent guess prefix: {partial if partial else '(empty)'}"
            "\nPredict the next uppercase letter:"
        )
        if self.use_chat_template:
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise RuntimeError(
                    f"Tokenizer for {self._model_name!r} does not expose apply_chat_template()."
                )
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=self.chat_generation_prompt,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_length,
                padding="max_length",
            )
        else:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_length,
                padding="max_length",
            )
        model_inputs = _filter_model_inputs(dict(enc), self._lm_forward_keys)
        model_inputs = {k: v.to(self.head.weight.device) for k, v in model_inputs.items()}
        attn = model_inputs.get("attention_mask")
        if self._lm_trainable:
            out = self.lm(**model_inputs, output_hidden_states=True, return_dict=True)
        else:
            with torch.no_grad():
                out = self.lm(**model_inputs, output_hidden_states=True, return_dict=True)
        last_hidden_state = _extract_last_hidden_state(out)
        if attn is not None:
            idx = _last_non_padding_index(attn)[0]
            h = last_hidden_state[0, idx]
        else:
            h = last_hidden_state[0, -1]
        h = h.to(self.char_head.weight.dtype)
        return self.char_head(h)

    def _sample_autoregressive_word(
        self,
        state: WordleState,
        deterministic: bool = False,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        constraints = self._feedback_constraints(state)
        letters: List[str] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for pos in range(5):
            logits = self._char_logits_for_partial(state, "".join(letters)).clone()
            # Hard position constraints from prior Wordle feedback.
            fixed = constraints["fixed"].get(pos)
            if fixed is not None:
                forced_idx = self._letter_to_idx(fixed)
                mask = torch.full_like(logits, float("-inf"))
                mask[forced_idx] = 0.0
                logits = logits + mask
            else:
                for blocked in constraints["blocked_pos"][pos]:
                    logits[self._letter_to_idx(blocked)] = float("-inf")
                for banned in constraints["banned_global"]:
                    logits[self._letter_to_idx(banned)] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            if deterministic:
                idx = int(torch.argmax(probs).item())
                lp = torch.log(probs[idx] + 1e-12)
                ent = torch.zeros((), device=probs.device, dtype=probs.dtype)
            else:
                a = dist.sample()
                idx = int(a.item())
                lp = dist.log_prob(a)
                ent = dist.entropy()
            letters.append(self._idx_to_letter(idx))
            log_probs.append(lp)
            entropies.append(ent)
        word = "".join(letters)
        if constraints["required"]:
            for req in constraints["required"]:
                if req not in word:
                    missing_idx = self._letter_to_idx(req)
                    # Replace earliest non-fixed spot to satisfy yellow/green evidence.
                    for i in range(5):
                        if constraints["fixed"].get(i) is None:
                            letters[i] = self._idx_to_letter(missing_idx)
                            break
                    word = "".join(letters)
                    if req in word:
                        break
        return word, torch.stack(log_probs).sum(), torch.stack(entropies).mean()

    def char_teacher_forcing_logits(self, state: WordleState, partial: str) -> torch.Tensor:
        """Return next-letter logits under teacher forcing prefix."""
        return self._char_logits_for_partial(state, partial)

    def expand_vocab(self, new_max_vocab_size: int) -> int:
        """Grow the action vocabulary (and head) to ``new_max_vocab_size`` words.

        The new word list is produced with the same priority + alphabetical-fill
        rule used at construction, so it is guaranteed to be an append-only
        extension of the current ``self.words``. Old head rows are copied
        verbatim into the new head, and only the appended rows are freshly
        initialized — so any prior ES / warm-start training on existing actions
        is preserved across curriculum stages. LoRA adapters and the LM body
        are untouched.

        Returns:
            The new ``self.action_dim``.
        """
        old_n = len(self.words)
        if new_max_vocab_size <= old_n:
            return self.action_dim

        new_words = _words_with_priority(self.vocab, new_max_vocab_size, self._priority)
        if new_words[:old_n] != self.words:
            raise RuntimeError(
                "expand_vocab: new word list is not an append-only extension of the "
                "current vocabulary. Did the priority list or underlying vocab change?"
            )
        new_n = len(new_words)
        if new_n == old_n:
            return self.action_dim

        weight = self.head.weight
        bias = self.head.bias
        in_features = weight.shape[1]
        device = weight.device
        dtype = weight.dtype

        new_head = nn.Linear(in_features, new_n).to(device=device, dtype=dtype)
        nn.init.orthogonal_(new_head.weight, gain=0.01)
        nn.init.zeros_(new_head.bias)
        with torch.no_grad():
            new_head.weight[:old_n].copy_(weight)
            if bias is not None and new_head.bias is not None:
                new_head.bias[:old_n].copy_(bias)

        self.head = new_head
        self.words = new_words
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for i, w in enumerate(self.words)}
        self.action_dim = new_n
        return self.action_dim

    def encode_prompt(self, state: WordleState) -> dict:
        text = _build_wordle_prompt(state, richer_prompt=self.richer_prompt)
        if self.use_chat_template:
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise RuntimeError(
                    f"Tokenizer for {self._model_name!r} does not expose apply_chat_template()."
                )
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ]
            enc = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=self.chat_generation_prompt,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_length,
                padding="max_length",
            )
        else:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_length,
                padding="max_length",
            )
        return _filter_model_inputs(dict(enc), self._lm_forward_keys)

    def forward_logits(self, state: WordleState) -> torch.Tensor:
        return self.forward_logits_batch([state])[0]

    def forward_logits_batch(self, states: Sequence[WordleState]) -> torch.Tensor:
        """Batched logits over a list of states. Returns ``[B, action_dim]``.

        All prompts are produced by ``encode_prompt`` (which already pads to
        ``max_prompt_length``), then concatenated along the batch dim and run
        through a single LM forward. The last non-padding hidden state of each
        row is gathered and passed through the shared head.
        """
        if not states:
            raise ValueError("forward_logits_batch requires at least one state.")
        device = self.head.weight.device
        encs = [self.encode_prompt(s) for s in states]
        keys = encs[0].keys()
        if not all(set(e.keys()) == set(keys) for e in encs):
            raise RuntimeError("encode_prompt returned inconsistent keys across states.")
        model_inputs = {
            k: torch.cat([e[k] for e in encs], dim=0).to(device) for k in keys
        }
        attn = model_inputs.get("attention_mask")
        if self._lm_trainable:
            out = self.lm(**model_inputs, output_hidden_states=True, return_dict=True)
        else:
            with torch.no_grad():
                out = self.lm(**model_inputs, output_hidden_states=True, return_dict=True)

        last_hidden_state = _extract_last_hidden_state(out)  # [B, T, H]
        if attn is not None:
            idx = _last_non_padding_index(attn)  # [B]
            batch_arange = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            h = last_hidden_state[batch_arange, idx]
        else:
            h = last_hidden_state[:, -1]
        h = h.to(self.head.weight.dtype)
        return self.head(h)

    def get_action(
        self,
        state_embedding: np.ndarray,
        deterministic: bool = False,
        previous_guesses: Optional[List[str]] = None,
        state: Optional[WordleState] = None,
    ) -> Tuple[int, Optional[torch.Tensor]]:
        if state is None:
            raise ValueError("WordleGPT2Policy.get_action requires `state=WordleState` (embedding is unused).")
        if self.action_granularity == "char":
            word, log_prob, _ = self._sample_autoregressive_word(state, deterministic=deterministic)
            action_idx = self.word_to_idx.get(word, 0)
            return action_idx, (None if deterministic else log_prob)
        logits = self.forward_logits(state)
        if previous_guesses:
            # Single clone before in-place masking: avoids a fresh allocation per
            # masked guess (was up to 5 clones per call on the last turn).
            logits = logits.clone()
            for g in previous_guesses:
                u = g.upper()
                if u in self.word_to_idx:
                    logits[self.word_to_idx[u]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action_idx = int(torch.argmax(probs).item())
            return action_idx, None
        dist = torch.distributions.Categorical(probs)
        action_idx = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor(action_idx, device=logits.device))
        return action_idx, log_prob

    def get_action_word(
        self,
        state_embedding: np.ndarray,
        deterministic: bool = False,
        previous_guesses: Optional[List[str]] = None,
        state: Optional[WordleState] = None,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        if state is None:
            raise ValueError("WordleGPT2Policy.get_action_word requires `state=WordleState`.")
        if self.action_granularity == "char":
            word, lp, _ = self._sample_autoregressive_word(state, deterministic=deterministic)
            return word, (None if deterministic else lp)
        idx, lp = self.get_action(
            state_embedding,
            deterministic=deterministic,
            previous_guesses=previous_guesses,
            state=state,
        )
        return self.idx_to_word[idx], lp

    def format_action_xml(
        self,
        state: WordleState,
        state_embedding: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        word, log_prob = self.get_action_word(
            state_embedding,
            deterministic=deterministic,
            previous_guesses=state.previous_guesses,
            state=state,
        )
        turn = state.turn_number + 1
        if turn == 1:
            think = f"Using the language model prior, opening with {word}."
        else:
            think = f"Conditioning on feedback, next guess: {word}."
        xml_action = f"<think>{think}</think>\n<guess>{word}</guess>"
        return xml_action, log_prob

    def sample_word_with_stats(
        self,
        state: WordleState,
        deterministic: bool = False,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Sample a guess word and return (word, log_prob, entropy)."""
        if self.action_granularity == "char":
            return self._sample_autoregressive_word(state, deterministic=deterministic)
        logits = self.forward_logits(state)
        if state.previous_guesses:
            logits = logits.clone()
            for g in state.previous_guesses:
                u = g.upper()
                if u in self.word_to_idx:
                    logits[self.word_to_idx[u]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if deterministic:
            idx = int(torch.argmax(probs).item())
            lp = torch.log(probs[idx] + 1e-12)
            ent = torch.zeros((), device=probs.device, dtype=probs.dtype)
        else:
            a = dist.sample()
            idx = int(a.item())
            lp = dist.log_prob(a)
            ent = dist.entropy()
        return self.idx_to_word[idx], lp, ent

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


WordleHFPolicy = WordleGPT2Policy
