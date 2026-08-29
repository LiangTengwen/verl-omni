"""OmniModelBase adapter for Qwen3.5-35B-A3B (Qwen3_5MoeForConditionalGeneration).

Qwen3.5-35B-A3B ships as a VLM: its config.json carries a ``vision_config``
(ViT, depth 27) plus image/video token ids, and this run trains it as such.  Without an adapter, ``OmniModelConfig.__post_init__`` raises
``NotImplementedError`` because the architecture is absent from verl-omni's
registry.

This adapter registers the architecture and removes the vision tower from the
FSDP training scope so its weights are neither sharded nor differentiated.
"""

import logging
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

logger = logging.getLogger(__name__)


@OmniModelBase.register("Qwen3_5MoeForConditionalGeneration", stage="thinker")
class Qwen35MoeThinkerAdapter(OmniModelBase):
    """verl-omni adapter for Qwen3.5-35B-A3B multimodal GRPO training.

    Mirrors the verified verl_npu ``main_ppo`` baseline: same checkpoint, same
    image-bearing dataset, vision tower trained rather than stripped.  The
    adapter exists only because ``OmniModelConfig.__post_init__`` requires the
    architecture to be registered; it deliberately changes no behaviour.
    """

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """Nothing is stripped: this is full multimodal training.

        The vision tower must stay in the module and in the FSDP graph, exactly
        as in the verified verl_npu ``main_ppo`` baseline
        (examples/grpo_trainer/run_qwen3_5_35b_grpo_910b_fsdp_noshuffle_nokl_new.sh),
        which trains this same checkpoint on this same image-bearing parquet.

        An earlier revision stripped ``model.visual`` and froze every parameter
        matching visual/vision/patch_embed.  Both were wrong for this goal:

        * Stripping made ``Qwen3_5MoeModel.forward`` raise ``AttributeError:
          'Qwen3_5MoeModel' object has no attribute 'visual'`` as soon as a
          batch carried ``pixel_values`` -- the vision branch is entered on
          ``pixel_values is not None`` (modeling_qwen3_5_moe.py:1848), which
          depends on the data, not on whether the tower exists.
        * Freezing (``requires_grad_(False)``) took those parameters out of
          FSDP2's sharding/all-gather management.  With
          ``fsdp_config.param_offload=True`` they then stayed on CPU, so
          ``self.pos_embed.weight.device`` reported ``cpu`` while the rest of
          the graph ran on ``npu:N`` -- the RuntimeError seen at
          modeling_qwen3_5_moe.py:1288.
        """
        return []

    @classmethod
    def configure_model(cls, module, model_config):
        """Apply the vision positional-embedding device fix, then defer to base.

        Nothing is stripped or frozen here (see ``get_strip_modules``).  The one
        change is a monkey-patch on ``Qwen3_5MoeVisionModel`` that makes the
        positional-embedding lookup take its device from the activation rather
        than from ``self.pos_embed.weight``.

        Why it is needed: verl turns on FSDP2 ``CPUOffloadPolicy`` for every
        ``forward_only`` engine (transformer_impl.py:435), which in GRPO means
        the **ref** model.  Under that policy the parameter reports ``cpu``
        while the module runs on ``npu:N``, so the original code builds its
        index tensor on the wrong device and ``F.embedding`` fails.  This is
        exactly why all 16 observed failures were in ``compute_ref_log_prob``
        and none in the actor.

        The patch is idempotent and numerically inert -- it only changes which
        device the indices are allocated on.
        """
        from verl_omni.models.transformers import apply_qwen3_5_vision_device_fix

        apply_qwen3_5_vision_device_fix()
        return super().configure_model(module, model_config)

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """Standard HF tokenizer -- Qwen3.5 needs no special chat-template wiring."""
        from verl.utils import hf_tokenizer

        return hf_tokenizer(
            model_path,
            trust_remote_code=getattr(model_config, "trust_remote_code", False),
        )

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """Return the real HF processor -- do NOT return None here.

        Qwen3.5-35B-A3B is a VLM checkpoint (config.json carries vision_config
        plus a preprocessor_config.json), so ``hf_processor`` returns a genuine
        ProcessorMixin for it.  ``hf_processor`` only returns None for
        *text-only* checkpoints, which this is not.

        Why this matters even for text-only GRPO
        ----------------------------------------
        ``OmniPPOTrainerSync._init_tokenizer`` (ray_omni_trainer.py) assigns
        ``self.processor = model_config.processor``, i.e. whatever this method
        returns, and hands it to the dataset.  ``RLHFDataset._build_messages``
        then asserts::

            assert self.processor is not None, "processor is needed to process multimodal data"

        That assert fires whenever a row has a non-empty ``images`` / ``videos``
        / ``audios`` column -- regardless of whether the *model* is being
        trained text-only.  Returning None here diverges from the verified
        verl_npu main_ppo baseline, which builds the processor via
        ``hf_processor(local_path)`` and therefore always had a real one.

        Keeping parity with that baseline is the point: same checkpoint, same
        processor, same dataset behaviour.
        """
        from verl.utils import hf_processor

        processor = hf_processor(
            model_path,
            trust_remote_code=getattr(model_config, "trust_remote_code", False),
            use_fast=True,
        )
        if processor is None:
            logger.warning(
                "hf_processor returned None for %s. If the dataset carries image/video/audio "
                "columns, RLHFDataset._build_messages will assert. Check that "
                "preprocessor_config.json exists in the checkpoint.",
                model_path,
            )
        return processor