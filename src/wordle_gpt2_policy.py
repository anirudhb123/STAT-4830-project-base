"""
Wordle policy using a pretrained Hugging Face causal LM plus a trainable linear head
over a fixed 5-letter vocabulary. Supports raw-text tokenization for GPT-2-style models
and chat-template tokenization for instruction-tuned models such as Gemma.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set, Union

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


class _WordleVocabTrie:
    """Trie over uppercase 5-letter words for online prefix-validity checks.

    Each node holds a length-26 list of child node indices (or None) and a terminal flag.
    A precomputed ``[num_nodes, 26]`` bool tensor (``_masks``) lets ``valid_children_mask``
    return a row in O(1) without rebuilding a tensor each call. The tensor is lazily moved
    to the requested device the first time it's accessed there, then cached per device.
    """

    __slots__ = ("_children", "_terminal", "_masks_cpu", "_masks_by_device")

    def __init__(self, words: Iterable[str]) -> None:
        # Node 0 is root. Children stored as list-of-lists for O(1) Python-side lookup.
        self._children: List[List[int]] = [[-1] * 26]
        self._terminal: List[bool] = [False]
        for raw in words:
            w = raw.upper()
            if len(w) != 5 or not w.isalpha():
                continue
            node = 0
            for ch in w:
                ci = ord(ch) - ord("A")
                nxt = self._children[node][ci]
                if nxt < 0:
                    self._children.append([-1] * 26)
                    self._terminal.append(False)
                    nxt = len(self._children) - 1
                    self._children[node][ci] = nxt
                node = nxt
            self._terminal[node] = True
        # Precompute the [num_nodes, 26] bool mask once on CPU; ship to GPU lazily.
        masks = torch.zeros((len(self._children), 26), dtype=torch.bool)
        for n, row in enumerate(self._children):
            for ci, c in enumerate(row):
                if c >= 0:
                    masks[n, ci] = True
        self._masks_cpu: torch.Tensor = masks
        self._masks_by_device: Dict[torch.device, torch.Tensor] = {masks.device: masks}

    def root(self) -> int:
        return 0

    def step(self, node: int, letter_idx: int) -> int:
        """Return the child node id for ``letter_idx``, or -1 if absent."""
        if node < 0:
            return -1
        return self._children[node][letter_idx]

    def _masks_on(self, device: torch.device) -> torch.Tensor:
        m = self._masks_by_device.get(device)
        if m is None:
            m = self._masks_cpu.to(device)
            self._masks_by_device[device] = m
        return m

    def valid_children_mask(self, node: int, device: torch.device) -> torch.Tensor:
        """bool[26] of letters that lead to at least one valid completion from ``node``."""
        if node < 0:
            return torch.zeros(26, dtype=torch.bool, device=device)
        return self._masks_on(device)[node]


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

        self.vocab_trie: Optional[_WordleVocabTrie] = (
            _WordleVocabTrie(self.words) if self.action_granularity == "char" else None
        )

        # In char mode the word-level head is unused at inference (action_idx is a
        # vocabulary lookup, not an argmax over logits) — keep it frozen so ES does
        # not waste ~hidden*action_dim perturbation dims on a parameter that has no
        # effect on rollout reward. Same reasoning applies to warm-start CE: the
        # char-mode path computes loss against char_head only.
        if self.action_granularity == "char":
            self.head.weight.requires_grad = False
            self.head.bias.requires_grad = False
        self._trie_step_count: int = 0
        self._trie_fallback_count: int = 0
        self._trie_oov_word_count: int = 0

    def reset_trie_stats(self) -> None:
        """Zero out trie instrumentation counters (call at start of each ES iter / WS epoch)."""
        self._trie_step_count = 0
        self._trie_fallback_count = 0
        self._trie_oov_word_count = 0

    def trie_stats(self) -> Dict[str, Union[int, float]]:
        """Snapshot of trie-mask usage stats; ``trie_fallback_rate`` is fallbacks / steps."""
        rate = (
            self._trie_fallback_count / self._trie_step_count
            if self._trie_step_count
            else 0.0
        )
        return {
            "trie_steps": self._trie_step_count,
            "trie_fallbacks": self._trie_fallback_count,
            "trie_fallback_rate": rate,
            "oov_words": self._trie_oov_word_count,
        }

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
        trie_node = self.vocab_trie.root() if self.vocab_trie is not None else -1
        for pos in range(5):
            logits = self._char_logits_for_partial(state, "".join(letters)).clone()
            device = logits.device

            # 1) Build feedback mask (bool[26], True == allowed by Wordle feedback).
            feedback_mask = torch.ones(26, dtype=torch.bool, device=device)
            fixed = constraints["fixed"].get(pos)
            if fixed is not None:
                feedback_mask[:] = False
                feedback_mask[self._letter_to_idx(fixed)] = True
            else:
                for blocked in constraints["blocked_pos"][pos]:
                    feedback_mask[self._letter_to_idx(blocked)] = False
                for banned in constraints["banned_global"]:
                    feedback_mask[self._letter_to_idx(banned)] = False

            # 2) Combine with trie mask if available; fall back to feedback-only on empty intersection.
            if self.vocab_trie is not None:
                trie_mask = self.vocab_trie.valid_children_mask(trie_node, device=device)
                combined = feedback_mask & trie_mask
                self._trie_step_count += 1
                if not bool(combined.any()):
                    self._trie_fallback_count += 1
                    combined = feedback_mask
            else:
                combined = feedback_mask

            # 3) Apply mask to logits (mask=False -> -inf).
            logits = logits.masked_fill(~combined, float("-inf"))

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

            # 4) Advance trie node; reset to root if fallback drove us off the tree.
            if self.vocab_trie is not None:
                trie_node = self.vocab_trie.step(trie_node, idx)
                if trie_node < 0:
                    trie_node = self.vocab_trie.root()
        word = self._post_sample_fixup(letters, constraints)
        if self.vocab_trie is not None and word not in self.word_to_idx:
            self._trie_oov_word_count += 1
        return word, torch.stack(log_probs).sum(), torch.stack(entropies).mean()

    def char_teacher_forcing_logits(self, state: WordleState, partial: str) -> torch.Tensor:
        """Return next-letter logits under teacher forcing prefix."""
        return self._char_logits_for_partial(state, partial)

    def _char_logits_for_partials_batch(
        self, states: Sequence[WordleState], partials: Sequence[str]
    ) -> torch.Tensor:
        """Batched next-letter logits across multiple (state, partial) pairs. Returns ``[B, 26]``."""
        if len(states) != len(partials):
            raise ValueError("states and partials must have the same length.")
        device = self.head.weight.device
        if not states:
            return torch.empty(0, 26, device=device, dtype=self.char_head.weight.dtype)

        encs: List[Dict[str, torch.Tensor]] = []
        for state, partial in zip(states, partials):
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
            encs.append(_filter_model_inputs(dict(enc), self._lm_forward_keys))

        keys = list(encs[0].keys())
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
        last_hidden_state = _extract_last_hidden_state(out)
        if attn is not None:
            idx = _last_non_padding_index(attn)
            batch_arange = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
            h = last_hidden_state[batch_arange, idx]
        else:
            h = last_hidden_state[:, -1]
        h = h.to(self.char_head.weight.dtype)
        return self.char_head(h)

    def sample_words_batch(
        self,
        states: Sequence[WordleState],
        deterministic: bool = False,
        avoid_previous_guesses: bool = True,
    ) -> List[str]:
        """Char-mode autoregressive sampler that batches the LM forward across ``states``.

        At each of the 5 character positions we run a single batched LM forward over all
        active games (one row per game), apply per-row feedback + trie masks, and sample.
        That is 5 LM forwards per turn instead of the (n_active * 5) the per-game path would
        do, which is the only way char-mode rollouts come close to the wall-clock of the
        word-level head on a GPU.

        ``avoid_previous_guesses=True`` performs a cheap post-sample fixup: if the sampled
        word matches one of ``state.previous_guesses``, we re-sample once with the same
        masks. This is best-effort — under near-deterministic policies the same word may
        recur — and is only meant to avoid trivially wasting a turn on a known guess.
        """
        if self.action_granularity != "char":
            raise RuntimeError(
                "sample_words_batch requires action_granularity='char'."
            )
        n = len(states)
        if n == 0:
            return []
        constraints_list = [self._feedback_constraints(s) for s in states]
        trie_nodes: List[int] = [
            self.vocab_trie.root() if self.vocab_trie is not None else -1 for _ in range(n)
        ]
        letters_per: List[List[str]] = [[] for _ in range(n)]

        for pos in range(5):
            partials = ["".join(letters_per[i]) for i in range(n)]
            logits_batch = self._char_logits_for_partials_batch(states, partials).clone()  # [n, 26]
            device = logits_batch.device

            for i in range(n):
                constraints = constraints_list[i]
                feedback_mask = torch.ones(26, dtype=torch.bool, device=device)
                fixed = constraints["fixed"].get(pos)
                if fixed is not None:
                    feedback_mask[:] = False
                    feedback_mask[self._letter_to_idx(fixed)] = True
                else:
                    for blocked in constraints["blocked_pos"][pos]:
                        feedback_mask[self._letter_to_idx(blocked)] = False
                    for banned in constraints["banned_global"]:
                        feedback_mask[self._letter_to_idx(banned)] = False

                if self.vocab_trie is not None:
                    trie_mask = self.vocab_trie.valid_children_mask(trie_nodes[i], device=device)
                    combined = feedback_mask & trie_mask
                    self._trie_step_count += 1
                    if not bool(combined.any()):
                        self._trie_fallback_count += 1
                        combined = feedback_mask
                else:
                    combined = feedback_mask
                logits_batch[i].masked_fill_(~combined, float("-inf"))

            probs = F.softmax(logits_batch, dim=-1)
            if deterministic:
                sampled = torch.argmax(probs, dim=-1)
            else:
                sampled = torch.distributions.Categorical(probs=probs).sample()

            for i in range(n):
                ci = int(sampled[i].item())
                letters_per[i].append(self._idx_to_letter(ci))
                if self.vocab_trie is not None:
                    nxt = self.vocab_trie.step(trie_nodes[i], ci)
                    trie_nodes[i] = nxt if nxt >= 0 else self.vocab_trie.root()

        words: List[str] = []
        for i in range(n):
            constraints = constraints_list[i]
            word = self._post_sample_fixup(letters_per[i], constraints)
            if (
                avoid_previous_guesses
                and word in (states[i].previous_guesses or [])
            ):
                alt = self._sample_alt_word(states[i], constraints, exclude={word}, deterministic=deterministic)
                if alt is not None:
                    word = alt
            if self.vocab_trie is not None and word not in self.word_to_idx:
                self._trie_oov_word_count += 1
            words.append(word)
        return words

    def _sample_alt_word(
        self,
        state: WordleState,
        constraints: Dict[str, Any],
        exclude: Set[str],
        deterministic: bool,
    ) -> Optional[str]:
        """Cheap fallback: pick the in-vocab word, consistent with feedback constraints,
        that has not been previously guessed and is not in ``exclude``. Used to dodge
        duplicate-guess turn waste when the LM samples a word the agent has already
        played. Returns None if no candidate exists.

        We skip the LM here on purpose: this is rare-path correction code, the trie
        already enforces in-vocab membership, and the candidate set is small (≤ vocab).
        """
        previous = set(state.previous_guesses or [])
        candidates: List[str] = []
        for w in self.words:
            if w in exclude or w in previous:
                continue
            # Defensive: skip any vocab entry that is not a clean 5-letter word.
            # Bad data (e.g. comment lines that leak through a loose filter) would
            # otherwise trigger KeyError on constraints["blocked_pos"][pos>=5].
            if len(w) != 5 or not w.isalpha():
                continue
            ok = True
            for pos, ch in enumerate(w):
                fixed = constraints["fixed"].get(pos)
                if fixed is not None and ch != fixed:
                    ok = False
                    break
                if ch in constraints["blocked_pos"][pos]:
                    ok = False
                    break
                if ch in constraints["banned_global"]:
                    ok = False
                    break
            if not ok:
                continue
            for req in constraints["required"]:
                if req not in w:
                    ok = False
                    break
            if ok:
                candidates.append(w)
        if not candidates:
            return None
        if deterministic:
            return candidates[0]
        # Random pick — cheap and avoids any LM call.
        import random as _random
        return _random.choice(candidates)

    def _post_sample_fixup(self, letters: List[str], constraints: Dict[str, Any]) -> str:
        """Apply the required-letter fixup, but only commit a swap if it yields an in-vocab word.

        The original fixup blindly overwrote a non-fixed position with a missing required
        letter, which often produced OOV words once the vocab-trie was active. We now try
        each non-fixed position in turn and keep the first swap whose resulting 5-letter
        word is in ``self.word_to_idx``. If no in-vocab swap exists, leave the word
        unchanged (caller will count it as OOV).
        """
        word = "".join(letters)
        if not constraints["required"]:
            return word
        for req in constraints["required"]:
            if req in word:
                continue
            missing_idx = self._letter_to_idx(req)
            best_swap: Optional[Tuple[int, str]] = None
            for i in range(5):
                if constraints["fixed"].get(i) is not None:
                    continue
                candidate = list(letters)
                candidate[i] = self._idx_to_letter(missing_idx)
                cand_word = "".join(candidate)
                if cand_word in self.word_to_idx:
                    best_swap = (i, cand_word)
                    break
            if best_swap is not None:
                pos, cand_word = best_swap
                letters[pos] = self._idx_to_letter(missing_idx)
                word = cand_word
            else:
                # Fallback: do the legacy overwrite (will count as OOV at the call site).
                for i in range(5):
                    if constraints["fixed"].get(i) is None:
                        letters[i] = self._idx_to_letter(missing_idx)
                        break
                word = "".join(letters)
        return word

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
        if self.action_granularity == "char":
            self.vocab_trie = _WordleVocabTrie(self.words)
            # The new head is fresh (Linear default requires_grad=True); refreeze.
            self.head.weight.requires_grad = False
            self.head.bias.requires_grad = False
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
