"""
Wordle policy using a frozen pretrained GPT-2–style model (e.g. DistilGPT-2) plus a
trainable linear head over a fixed 5-letter vocabulary. Intended for Evolution Strategies
on the head only while the transformer stays fixed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

try:
    from .wordle_env import WordVocabulary, WordleState, MOCK_WORDLE_TARGETS
except ImportError:
    from wordle_env import WordVocabulary, WordleState, MOCK_WORDLE_TARGETS


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


def _build_wordle_prompt(state: WordleState) -> str:
    """Turn game state into a short natural-language prompt for the LM."""
    lines = [
        "You are playing Wordle. Guess a valid 5-letter English word.",
        f"Turn {state.turn_number + 1} of 6.",
    ]
    if state.previous_guesses:
        lines.append("Previous guesses and feedback:")
        for g, fb in zip(state.previous_guesses, state.feedback_history):
            lines.append(f"- {g}: {fb}")
    else:
        lines.append("No guesses yet. Choose a strong opening word.")
    lines.append("Reply with exactly one 5-letter uppercase word as your next guess.")
    return "\n".join(lines)


class WordleGPT2Policy(nn.Module):
    """
    Frozen GPT-2 / DistilGPT-2 encoder + trainable logits head over a vocabulary subset.

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
    ):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "wordle_gpt2_policy requires `transformers`. Install with: pip install transformers"
            )

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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModel.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        for p in self.lm.parameters():
            p.requires_grad = False
        self.lm.eval()

        hidden = self.lm.config.hidden_size
        self.head = nn.Linear(hidden, self.action_dim)
        self.max_prompt_length = max_prompt_length
        self._model_name = model_name

        self.apply(self._init_head)

    def _init_head(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)

    def encode_prompt(self, state: WordleState) -> dict:
        text = _build_wordle_prompt(state)
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
