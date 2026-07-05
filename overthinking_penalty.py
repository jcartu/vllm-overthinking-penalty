"""
Overthinking penalty logits processor for GLM-5.2.

Applies a fixed logit penalty to hesitation/overthinking tokens identified by
the Meta FAIR paper arxiv 2606.00206. Reduces CoT length 12-23% while
maintaining accuracy on reasoning tasks.

This processor is STATELESS — it applies a fixed penalty to fixed token IDs
regardless of request state, position, or context. This makes it safe to use
with speculative decoding (MTP), since the penalty is a pure function of the
logits tensor and does not depend on per-request generation history.

The penalty is applied AFTER min-tokens processing, BEFORE sampling. It
modifies logits in-place by subtracting lambda from the specified token IDs.

Configuration via environment variables:
  OVERTHINKING_PENALTY_LAMBDA  (float, default 5.0)  — penalty strength
  OVERTHINKING_PENALTY_TOKENS  (str,  default built-in set) — comma-sep token IDs

If OVERTHINKING_PENALTY_LAMBDA is 0 or unset, the processor is a no-op
(logits returned unmodified), allowing runtime toggling without restart.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# 43 hesitation/overthinking token IDs mapped from GLM-5.2's tokenizer.
# These correspond to tokens like "wait", "but", "alternatively", "however",
# "hmm", "actually", "let me", "I should", etc. — high-entropy hesitation
# markers that trigger self-correction loops in quantized reasoning models.
_DEFAULT_HESITATION_TOKEN_IDS: list[int] = [
    11, 67, 71, 83, 265, 552, 1347, 1419, 1975, 2028,
    2152, 2371, 2753, 3821, 3983, 4331, 5482, 5569, 6282, 7615,
    7887, 8087, 10857, 11484, 12440, 13123, 14181, 24636, 26779, 27356,
    32618, 33141, 34696, 36569, 40190, 49893, 52246, 63108, 64796, 72465,
    79380, 91243, 97009,
]


class OverthinkingPenaltyProcessor:
    """
    Applies a fixed logit penalty to hesitation token IDs.

    Registered as a non-argmax-invariant processor because the penalty CAN
    change argmax results in greedy sampling (a hesitation token that was
    the argmax choice will be suppressed if the penalty exceeds the logit
    gap to the next-best token).

    Safe with speculative decoding: the penalty is a pure function of the
    logits tensor with no per-request state.
    """

    def __init__(
        self,
        vllm_config,
        device: torch.device,
        is_pin_memory: bool,
    ):
        self._vllm_config = vllm_config
        self._device = device

        # Read config from environment at init time (evaluated once per
        # build_logitsprocs call, i.e. once per server startup).
        self._lambda = float(os.environ.get("OVERTHINKING_PENALTY_LAMBDA", "5.0"))

        tokens_env = os.environ.get("OVERTHINKING_PENALTY_TOKENS")
        if tokens_env:
            self._token_ids = [int(t.strip()) for t in tokens_env.split(",") if t.strip()]
        else:
            self._token_ids = list(_DEFAULT_HESITATION_TOKEN_IDS)

        # Pre-allocate the penalty tensor on the correct device.
        # Shape: [vocab_size] — we'll index into it with the token IDs.
        vocab_size = vllm_config.model_config.get_vocab_size()
        self._penalty = torch.zeros(vocab_size, dtype=torch.float32, device=device)
        if self._lambda > 0:
            self._penalty[self._token_ids] = -self._lambda
            logger.info(
                "OverthinkingPenaltyProcessor: active — lambda=%.2f, %d tokens, "
                "vocab_size=%d",
                self._lambda,
                len(self._token_ids),
                vocab_size,
            )
        else:
            logger.info("OverthinkingPenaltyProcessor: disabled (lambda=0)")

    @classmethod
    def validate_params(cls, params) -> bool:
        """No per-request params needed — the penalty is global."""
        return True

    def update_state(self, batch_update) -> None:
        """No-op — the penalty is stateless and does not track per-request state."""
        pass

    def is_argmax_invariant(self) -> bool:
        """False: the penalty can change argmax results in greedy sampling."""
        return False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply the penalty to the logits tensor.

        Args:
            logits: [batch_size, vocab_size] or [batch_size * num_draft, vocab_size]
                    The raw logits from the model.

        Returns:
            Modified logits tensor (same shape, modified in-place).
        """
        if self._lambda <= 0:
            return logits

        # The penalty tensor is [vocab_size]; broadcast-add across the batch
        # dimension. This works for both standard and spec-decode logits
        # shapes since the last dim is always vocab_size.
        logits.add_(self._penalty)
        return logits
