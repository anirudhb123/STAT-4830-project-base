"""
Generation-based Wordle policy for week16+. Wraps a HF causal LM
(default: ``PrimeIntellect/Qwen3-1.7B-Wordle-SFT``) with LoRA, produces guesses
by calling ``model.generate(...)`` and parsing ``<guess>WORD</guess>`` from the
output. Designed to plug into the existing ES infrastructure in
``src/es_wordle.py`` without modifying that module.

Differences vs ``WordleGPT2Policy``:
- No ``char_head`` / fresh letter classifier; Qwen's pretrained ``lm_head`` produces
  the actual generation distribution.
- No ``vocab_trie`` and no per-position feedback masks; the SFT'd model is trusted
  to emit valid Wordle words and follow feedback. Bad emissions waste a turn at
  zero reward (env-level penalty).
- Action space is whatever the LM tokenizer produces; the trainable parameters
  are the LoRA adapters only.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

try:
    from .wordle_env import WordleState
    from .wordle_hints import build_constraint_summary
except ImportError:
    from wordle_env import WordleState
    from wordle_hints import build_constraint_summary


_GUESS_RE = re.compile(r"<guess>(.*?)</guess>", re.IGNORECASE | re.DOTALL)
_FIVE_LETTER_RE = re.compile(r"[A-Za-z]{5}")
_NON_ALPHA_RE = re.compile(r"[^A-Za-z]")
# Strip both closed and unclosed-tail thinking tags before the 5-letter-run fallback.
# Without this the literal word "think" inside ``<think>...`` is the first 5-letter
# match and gets emitted as the guess on every turn.
_THINK_CLOSED_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED_TAIL_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"</?\s*[A-Za-z][A-Za-z0-9_-]*\s*>")
# Common English 5-grams that appear in “reasoning” text but are useless as Wordle guesses
# when they beat a real word as the *first* _FIVE_LETTER_RE match.
_FALLBACK_SKIP: frozenset[str] = frozenset(
    {
        "ABOUT",
        "AFTER",
        "AGAIN",
        "BEING",
        "COULD",
        "EVERY",
        "FIRST",
        "GUESS",
        "HAVEN",
        "MONTH",
        "NEVER",
        "OTHER",
        "PLANS",
        "PLAYI",  # “playing” fragment when tokenizer splits oddly — rarely 5
        "SHALL",
        "SHOULD",
        "SINCE",
        "STILL",
        "THANK",
        "THEIR",
        "THERE",
        "THESE",
        "THING",
        "THINK",
        "THOSE",
        "THREE",
        "UNDER",
        "UNTIL",
        "WHATS",
        "WHERE",
        "WHICH",
        "WOULD",
        "WRITE",
    }
)


def _build_wordle_prompt(state: WordleState, richer_prompt: bool = True) -> str:
    """Plain-text Wordle prompt. Mirrors ``wordle_gpt2_policy._build_wordle_prompt``;
    duplicated here to avoid an import coupling between policies."""
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


def _load_canonical_answers() -> List[str]:
    """Read the bundled NYT 2,315-answer pool from ``data/wordle_answers.txt``."""
    here = Path(__file__).resolve().parent
    answers_path = here.parent / "data" / "wordle_answers.txt"
    out: List[str] = []
    with answers_path.open("r", encoding="utf-8") as f:
        for raw in f:
            w = raw.strip().upper()
            if not w or w.startswith("#"):
                continue
            if len(w) == 5 and w.isalpha():
                out.append(w)
    return out


class WordleQwenPolicy(nn.Module):
    """Generation-based Wordle policy.

    Plugs into the existing ES infrastructure in ``es_wordle.py``:

    - Declares ``action_granularity="char"`` so ES rollouts route through
      ``sample_words_batch`` (see ``es_wordle._rollout_batched``).
    - Defines ``forward_logits_batch`` as a zeros stub. ES's
      ``_evaluate_perturbation`` (``es_wordle.py`` line 217), the per-step probe,
      and the periodic eval all check ``hasattr(policy, "forward_logits_batch")``
      to gate the *batched* path; without it they silently fall back to per-state
      serial loops that call ``format_action_xml`` once per state -- which here
      means one ``model.generate`` call per game and is catastrophically slow.
      The actual zeros are never consumed because the char-mode branch supplies
      words via ``sample_words_batch``.
    - Repurposes the ``trie_stats`` / ``reset_trie_stats`` hooks to surface a
      parse-failure rate as the ``fb%`` column in the ES train log, so the same
      diagnostics surface as in week14 with a different meaning.
    """

    def __init__(
        self,
        model_name: str = "PrimeIntellect/Qwen3-1.7B-Wordle-SFT",
        max_prompt_length: int = 512,
        enable_thinking: bool = False,
        max_new_tokens: int = 64,
        gen_temperature: float = 0.8,
        gen_top_p: float = 0.9,
        use_lora: bool = True,
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.05,
        cast_lora_to_fp32: bool = True,
        richer_prompt: bool = True,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "wordle_qwen_policy requires `transformers`. Install: pip install transformers"
            )
        if use_lora and not _PEFT_AVAILABLE:
            raise ImportError("LoRA requires `peft`. Install: pip install peft")

        self._model_name = model_name
        self.max_prompt_length = max_prompt_length
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.richer_prompt = richer_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **(tokenizer_kwargs or {}))
        # HF generate needs left-padding for batched mixed-length prompts.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lm = AutoModelForCausalLM.from_pretrained(model_name, **(model_kwargs or {}))
        self._lm_forward_keys = tuple(inspect.signature(self.lm.forward).parameters.keys())

        # Whether thinking was actually accepted by apply_chat_template; flipped
        # to False on TypeError fallback below.
        self._enable_thinking_supported = True

        self._lm_trainable = False
        if use_lora:
            target_modules = (
                list(lora_target_modules)
                if lora_target_modules is not None
                else ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.lm = get_peft_model(self.lm, peft_config)
            self._lm_trainable = True
            if cast_lora_to_fp32:
                # ES (es_wordle.py:626) flattens trainable params using their native
                # dtype. With a bf16 base, LoRA inits in bf16 and ES gradient math
                # then runs in bf16 -- too coarse for antithetic perturbations to
                # extract a clean signal at small population sizes. Cast LoRA to
                # fp32; the bf16 base remains frozen and PEFT handles the dtype mix.
                for p in self.lm.parameters():
                    if p.requires_grad:
                        p.data = p.data.float()
        else:
            for p in self.lm.parameters():
                p.requires_grad = False

        # Index-lookup vocab. ES char-mode never reads this for logits, but
        # ``trie_stats`` / OOV diagnostics + downstream eval helpers index into it.
        self.words: List[str] = _load_canonical_answers()
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_word = {i: w for i, w in enumerate(self.words)}
        self.action_dim = len(self.words)

        # Routes ES through ``sample_words_batch`` in ``_rollout_batched``.
        self.action_granularity = "char"

        # Diagnostics counters; reset once per ES iter via ``reset_trie_stats``.
        self._parse_attempts: int = 0
        self._parse_failures: int = 0
        self._oov_words: int = 0

    # ----- ES interface -----

    def reset_trie_stats(self) -> None:
        """ES iteration boundary hook. Repurposed: clears parse-failure counters."""
        self._parse_attempts = 0
        self._parse_failures = 0
        self._oov_words = 0

    def trie_stats(self) -> Dict[str, Any]:
        """Reported as the ``fb%`` column in ES verbose output. Repurposed here:
        ``trie_fallback_rate`` is the parse-failure rate (fraction of generations
        where ``<guess>5-letter</guess>`` was not cleanly extractable), and
        ``oov_words`` counts words emitted that aren't in the 2,315 answer pool."""
        rate = (
            self._parse_failures / self._parse_attempts
            if self._parse_attempts > 0
            else float("nan")
        )
        return {
            "trie_steps": int(self._parse_attempts),
            "trie_fallbacks": int(self._parse_failures),
            "trie_fallback_rate": float(rate) if rate == rate else float("nan"),
            "oov_words": int(self._oov_words),
        }

    def forward_logits_batch(self, states: Sequence[WordleState]) -> torch.Tensor:
        """Zeros stub. Required for the ``hasattr`` gate in ``es_wordle.py`` that
        switches between batched and serial perturbation/eval/probe paths.
        The actual values are never consumed because ``action_granularity="char"``
        routes word selection through ``sample_words_batch``."""
        device = next(self.parameters()).device
        return torch.zeros(len(states), self.action_dim, device=device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----- Generation + parsing -----

    def _build_chat_inputs(self, states: Sequence[WordleState]) -> Dict[str, torch.Tensor]:
        """Apply chat template + left-pad across the batch. Returns a dict of
        tensors ready for ``model.generate``."""
        messages_batch = [
            [{"role": "user", "content": _build_wordle_prompt(s, richer_prompt=self.richer_prompt)}]
            for s in states
        ]
        kwargs: Dict[str, Any] = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        if self._enable_thinking_supported:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            enc = self.tokenizer.apply_chat_template(messages_batch, **kwargs)
        except TypeError:
            # Older transformers/tokenizer versions reject ``enable_thinking``.
            self._enable_thinking_supported = False
            kwargs.pop("enable_thinking", None)
            enc = self.tokenizer.apply_chat_template(messages_batch, **kwargs)
        device = next(self.parameters()).device
        return {k: v.to(device) for k, v in enc.items()}

    def _parse_word_from_text(self, text: str) -> Tuple[str, bool]:
        """Layered fallback parser. Returns ``(word, parsed_clean)``.

        - ``parsed_clean=True`` iff a well-formed ``<guess>5-letter-word</guess>``
          was found (after stripping non-alpha chars like brackets/whitespace).
        - ``parsed_clean=False`` indicates a fallback path (first 5-letter run in
          the post-thinking text, or the ``"XXXXX"`` sentinel).
        """
        # 1. Last <guess>...</guess> match (handles model re-thinks mid-output).
        matches = _GUESS_RE.findall(text)
        if matches:
            cand = _NON_ALPHA_RE.sub("", matches[-1]).upper()
            if len(cand) == 5 and cand.isalpha():
                return cand, True
        # 2. First 5-letter alphabetic run *outside* any <think> block. Strip
        #    closed thinking blocks first, then drop any unclosed tail (the
        #    model was truncated mid-thought). Also strip residual XML-ish tags
        #    so the literal word inside e.g. ``<answer>`` isn't part of a 5-run
        #    that bridges letters and tag punctuation. Without these passes the
        #    word ``think`` itself becomes the guess, which is the failure mode
        #    observed on the SFT/RL Wordle checkpoints when ``max_new_tokens``
        #    is too small to fit a complete thinking trace + ``<guess>`` tag.
        cleaned = _THINK_CLOSED_RE.sub(" ", text)
        cleaned = _THINK_UNCLOSED_TAIL_RE.sub(" ", cleaned)
        cleaned = _TAG_RE.sub(" ", cleaned)
        runs = _FIVE_LETTER_RE.findall(cleaned)
        if runs:
            upper_runs = [r.upper() for r in runs]
            # Prefer a 5-letter span that is an actual Wordle solution word
            # (reversed first — models often state the proposed guess last).
            for r in reversed(upper_runs):
                if r in self.word_to_idx:
                    return r, False
            for r in upper_runs:
                if r in self.word_to_idx:
                    return r, False
            # Skip common reasoning fragments; then prefer a later token (guess-like).
            for r in reversed(upper_runs):
                if r not in _FALLBACK_SKIP:
                    return r, False
            for r in upper_runs:
                if r not in _FALLBACK_SKIP:
                    return r, False
            return upper_runs[-1], False
        # 3. Sentinel. Env returns "Invalid guess", reward=0, turn consumed -- ES
        #    sees this perturbation as low-fitness and learns to avoid it.
        return "XXXXX", False

    @torch.no_grad()
    def sample_words_batch(
        self,
        states: Sequence[WordleState],
        deterministic: bool = False,
        avoid_previous_guesses: bool = True,
    ) -> List[str]:
        """Main rollout path. Builds prompts, runs ``model.generate`` once over the
        batch, decodes, parses ``<guess>WORD</guess>``. ES char-mode dispatches
        here from ``_rollout_batched``."""
        if not states:
            return []

        enc = self._build_chat_inputs(states)
        input_len = int(enc["input_ids"].shape[1])

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=not deterministic,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if not deterministic:
            gen_kwargs["temperature"] = self.gen_temperature
            gen_kwargs["top_p"] = self.gen_top_p

        was_training = self.lm.training
        self.lm.eval()
        try:
            out = self.lm.generate(**enc, **gen_kwargs)
        finally:
            if was_training:
                self.lm.train()

        gen_ids = out[:, input_len:]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        words: List[str] = []
        for i, text in enumerate(texts):
            self._parse_attempts += 1
            word, parsed_clean = self._parse_word_from_text(text)
            if not parsed_clean:
                self._parse_failures += 1
            if word not in self.word_to_idx:
                self._oov_words += 1
            if avoid_previous_guesses and word in (states[i].previous_guesses or []):
                # Rare-path fallback: pick the first answer-pool word the agent
                # hasn't tried yet. A well-trained SFT shouldn't need this.
                alt = next(
                    (w for w in self.words if w not in (states[i].previous_guesses or [])),
                    None,
                )
                if alt is not None:
                    word = alt
            words.append(word)
        return words

    def format_action_xml(
        self,
        state: WordleState,
        state_embedding: Any,
        deterministic: bool = False,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """Single-state path. Used by ES eval/probe fallbacks that don't take the
        batched route. Wraps ``sample_words_batch`` so generation logic lives in
        one place."""
        word = self.sample_words_batch([state], deterministic=deterministic)[0]
        turn = state.turn_number + 1
        if turn == 1:
            think = f"Using SFT prior, opening with {word}."
        else:
            think = f"Conditioning on feedback, next guess: {word}."
        xml = f"<think>{think}</think>\n<guess>{word}</guess>"
        return xml, None

    def get_action(
        self,
        state_embedding: Any,
        deterministic: bool = False,
        previous_guesses: Optional[List[str]] = None,
        state: Optional[WordleState] = None,
    ) -> Tuple[int, Optional[torch.Tensor]]:
        """Compatibility shim for any caller that expects the GPT-2 policy's
        ``get_action`` interface. ES does not use this when ``forward_logits_batch``
        is present, but a few helper functions do."""
        if state is None:
            raise ValueError("WordleQwenPolicy.get_action requires `state=WordleState`.")
        word = self.sample_words_batch([state], deterministic=deterministic)[0]
        return self.word_to_idx.get(word, 0), None
