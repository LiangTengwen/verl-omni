"""Qwen3.5-MoE (thinker-only) rollout pipeline adapter.

Provides the per-stage pipeline topology for Qwen3.5-35B-A3B by delegating to
vLLM-Omni's frozen pipeline definition — no duplication of what vLLM-Omni
already owns.

Prerequisite
------------
``vllm_omni.model_executor.models.qwen3_5_moe.pipeline`` must exist in the
installed vLLM-Omni.  It is supplied by the Qwen3.5-MoE patch (4 files, 72
lines) documented in the implementation plan; without it the import below
raises ``ModuleNotFoundError`` at ``import verl_omni`` time.

Differences from the Qwen3-Omni adapter
---------------------------------------
* Only one pipeline mode exists (``thinker_only``).  Qwen3.5-MoE has no
  talker / code2wav stages, so there is nothing to select between.
* No ``enable_audio_output`` HF override: that field belongs to Qwen3-Omni's
  nested config and is absent from Qwen3.5's flat config.
* ``model_arch`` is taken from the pipeline definition rather than hardcoded —
  Qwen3.5-MoE has no separate thinker-only model class, the pipeline loads
  upstream vLLM's ``Qwen3_5MoeForConditionalGeneration`` directly.
"""

from vllm_omni.config.pipeline_registry import register_pipeline
from vllm_omni.model_executor.models.qwen3_5_moe.pipeline import (
    QWEN3_5_MOE_THINKER_ONLY_PIPELINE,
)

from verl.utils.device import is_npu_available

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase


@OmniRolloutPipelineBase.register("qwen3_5_moe")
class Qwen35MoeRolloutAdapter(OmniRolloutPipelineBase):
    """Rollout pipeline topology adapter for Qwen3.5-35B-A3B.

    Registered under ``model_type="qwen3_5_moe"`` — this is the value passed as
    ``+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name``.  Note
    that the registration key and the vLLM-Omni pipeline id differ:

    ==========================  ===============================
    verl-omni registration key  ``qwen3_5_moe``
    vLLM-Omni pipeline id       ``qwen3_5_moe_thinker_only``
    ==========================  ===============================

    which is why :meth:`get_pipeline_id` must be overridden — the base
    implementation returns the registration key, and the generated deploy YAML
    needs the pipeline id.
    """

    @classmethod
    def build_stage_configs(cls, pipeline_mode="thinker_only"):
        """Return per-stage pipeline topology objects for Qwen3.5-MoE.

        Args:
            pipeline_mode (str): Pipeline mode selector.  Only ``thinker_only``
                is supported.

        Returns:
            list: Single-element list holding vLLM-Omni's stage 0 topology.
        """
        if pipeline_mode != "thinker_only":
            raise ValueError(
                f"Unknown pipeline_mode={pipeline_mode!r} for qwen3_5_moe. "
                "Only 'thinker_only' is supported: Qwen3.5-MoE has no talker/code2wav stages."
            )
        stages = list(QWEN3_5_MOE_THINKER_ONLY_PIPELINE.stages)
        # Guard against upstream changes that silently add stages.
        assert len(stages) == 1, (
            f"Expected 1 stage in qwen3_5_moe thinker-only pipeline, got {len(stages)}. "
            "vLLM-Omni may have changed the pipeline definition."
        )
        return stages

    @classmethod
    def rollout_flags(cls, pipeline_mode="thinker_only"):
        """No inter-stage flags: a single-stage pipeline has no 'between'."""
        return {}

    @classmethod
    def get_pipeline_id(cls, pipeline_mode: str = "thinker_only") -> str:
        """Return the vLLM-Omni pipeline model_type (``qwen3_5_moe_thinker_only``)."""
        return QWEN3_5_MOE_THINKER_ONLY_PIPELINE.model_type

    @classmethod
    def ensure_pipeline_registered(cls, pipeline_mode: str = "thinker_only") -> None:
        """Register the thinker-only pipeline in vLLM-Omni's registry.

        Idempotent in practice: the installed vLLM-Omni already carries this
        pipeline in ``OMNI_PIPELINES``, but the rollout replica may run before
        that module has been imported, so register explicitly.
        """
        register_pipeline(QWEN3_5_MOE_THINKER_ONLY_PIPELINE)

    @classmethod
    def get_engine_hf_overrides(cls, pipeline_mode: str = "thinker_only") -> dict:
        """No HF config overrides.

        Qwen3.5's config.json is flat (``text_config`` / ``vision_config`` at
        the root, no ``thinker_config``), and it has no audio-output switch to
        turn off, so nothing needs overriding here.
        """
        return {}

    @classmethod
    def get_stage_engine_extras(cls, stage_id: int, pipeline_mode: str = "thinker_only") -> dict:
        """Return stage-0 engine extras: model arch plus the NPU weight-format fix."""
        if stage_id != 0:
            return {}
        extras = {"model_arch": QWEN3_5_MOE_THINKER_ONLY_PIPELINE.model_arch}
        if is_npu_available:
            # vllm-ascend rejects wake_up() under FRACTAL_NZ: reordered weights
            # lose precision across RL weight syncs.  This dict is promoted to
            # the top-level `additional_config` by _write_deploy_config, because
            # vllm-ascend reads it from a per-TP-worker global singleton that is
            # built from the CLI arg, not from stage-level engine_extras.
            extras["additional_config"] = {"weight_nz_mode": 0}
        return extras
