"""
Wordle policy using a pretrained GPT-2–style model (e.g. GPT-2 / DistilGPT-2) plus a
trainable linear head over a fixed 5-letter vocabulary. Optional PEFT LoRA on the last
transformer blocks; otherwise the backbone is frozen and only the head is trained (e.g. ES).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

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
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoModel = None  # type: ignore
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


class WordleGPT2Policy(nn.Module):
    """
    GPT-2 / DistilGPT-2 encoder + trainable logits head over a vocabulary subset.
    Optional LoRA on the last few transformer blocks (`use_lora=True`); otherwise the
    backbone is frozen.

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
        use_lora: bool = False,
        lora_r: int = 4,
        lora_alpha: float = 16.0,
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
        words = _words_with_priority(full_vocab, max_vocab_size, priority)

        self.words: List[str] = words
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for i, w in enumerate(self.words)}
        self.action_dim = len(self.words)
        # Alias for code that expects `policy.vocab`
        self.vocab = full_vocab

        self.richer_prompt = richer_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModel.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._lm_trainable = False
        if use_lora:
            # LoRA on all blocks' c_attn / c_proj (Conv1D). layers_to_transform is omitted
            # because several PEFT + GPT-2 builds fail to match targets when it is set.
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=int(lora_alpha),
                target_modules=["c_attn", "c_proj"],
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

        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.zeros_(self.head.bias)

    def encode_prompt(self, state: WordleState) -> dict:
        text = _build_wordle_prompt(state, richer_prompt=self.richer_prompt)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length,
            padding="max_length",
        )
        return {k: v for k, v in enc.items()}

    def forward_logits(self, state: WordleState) -> torch.Tensor:
        device = self.head.weight.device
        enc = self.encode_prompt(state)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)

        if self._lm_trainable:
            out = self.lm(input_ids=input_ids, attention_mask=attn)
        else:
            with torch.no_grad():
                out = self.lm(input_ids=input_ids, attention_mask=attn)
        # Last non-padding position for the single sequence
        if attn is not None:
            seq_lens = attn.sum(dim=1).long() - 1
            idx = seq_lens.clamp(min=0)
            h = out.last_hidden_state[0, idx[0]]
        else:
            h = out.last_hidden_state[0, -1]
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
        logits = self.forward_logits(state)
        if previous_guesses:
            for g in previous_guesses:
                u = g.upper()
                if u in self.word_to_idx:
                    logits = logits.clone()
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

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
