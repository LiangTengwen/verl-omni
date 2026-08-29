"""Device fix for Qwen3.5 vision positional embedding under FSDP2 CPU offload.

Problem
-------
``Qwen3_5MoeVisionModel.fast_pos_embed_interpolate`` picks its device from the
parameter it is about to index::

    device = self.pos_embed.weight.device
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]

Under FSDP2 with ``CPUOffloadPolicy`` -- which verl enables unconditionally for
any ``forward_only`` engine (transformer_impl.py:435, i.e. the **ref** model in
GRPO) -- ``.weight.device`` reports ``cpu`` while the module is executing on
``npu:N``.  The indices are then built on CPU and ``F.embedding`` raises::

    RuntimeError: Expected all tensors to be on the same device, but got
    indices is on cpu, different from other tensors on npu:N

This is why every failure was in ``compute_ref_log_prob`` and never in the
actor: only the ref engine sets ``forward_only=True``.

Fix
---
Derive the device from the *activation* rather than from the parameter.  The
caller (``Qwen3_5MoeVisionModel.forward``) has already run ``self.patch_embed``,
so ``hidden_states`` is guaranteed to be on the compute device.  We thread that
device in via a wrapper on ``forward`` and let
``fast_pos_embed_interpolate`` prefer it.

Scope
-----
Only the device *selection* changes; all arithmetic is untouched, so numerics
are identical to upstream.  This is a no-op when the parameter already reports
the correct device (i.e. when CPU offload is not in play).
"""

import logging

logger = logging.getLogger(__name__)

_PATCH_FLAG = "_verl_omni_device_patched"


def apply_qwen3_5_vision_device_fix() -> bool:
    """Patch ``Qwen3_5MoeVisionModel`` so positional-embedding lookup is device-safe.

    Returns:
        bool: True if the patch was applied (or already present), False if the
        target class could not be imported (non-Qwen3.5 transformers build).
    """
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel
    except ImportError as exc:
        logger.info("Qwen3.5 vision device fix skipped: %s", exc)
        return False

    if getattr(Qwen3_5MoeVisionModel, _PATCH_FLAG, False):
        return True

    original_interpolate = Qwen3_5MoeVisionModel.fast_pos_embed_interpolate
    original_forward = Qwen3_5MoeVisionModel.forward

    def _patched_interpolate(self, grid_thw):
        # Set by _patched_forward for the duration of one forward pass.
        target_device = getattr(self, "_verl_omni_compute_device", None)
        if target_device is None:
            return original_interpolate(self, grid_thw)

        pos_embed = self.pos_embed
        if pos_embed.weight.device == target_device:
            # Nothing to correct -- offload is not in play on this rank.
            return original_interpolate(self, grid_thw)

        # Temporarily present the embedding on the compute device so the
        # original implementation's `device = self.pos_embed.weight.device`
        # resolves correctly. Restore afterwards so FSDP's offload bookkeeping
        # (which owns the CPU copy) is left exactly as it was.
        self.pos_embed = pos_embed.to(target_device)
        try:
            return original_interpolate(self, grid_thw)
        finally:
            self.pos_embed = pos_embed

    def _patched_forward(self, hidden_states, grid_thw, *args, **kwargs):
        # hidden_states has already been produced on the compute device by the
        # caller, so it is a trustworthy source of truth -- unlike a parameter
        # that FSDP may have offloaded.
        self._verl_omni_compute_device = hidden_states.device
        try:
            return original_forward(self, hidden_states, grid_thw, *args, **kwargs)
        finally:
            self._verl_omni_compute_device = None

    Qwen3_5MoeVisionModel.fast_pos_embed_interpolate = _patched_interpolate
    Qwen3_5MoeVisionModel.forward = _patched_forward
    setattr(Qwen3_5MoeVisionModel, _PATCH_FLAG, True)

    logger.info("Applied Qwen3.5 vision positional-embedding device fix (FSDP2 CPU offload).")
    return True


__all__ = ["apply_qwen3_5_vision_device_fix"]