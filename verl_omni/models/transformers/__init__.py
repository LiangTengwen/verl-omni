# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DEPRECATED: This package will be removed in v0.3.0.
# The legacy Qwen3-Omni monkey-patches are no longer needed with the V1 trainer
# (verl_omni.trainer.main_omni). Please see run_qwen3_omni_thinker_gspo_lora_v1.sh
# for the V1 migration path.

# Exception: qwen3_5_moe_vision is NOT legacy. Qwen3.5-35B-A3B's vision tower
# still needs a device fix under FSDP2 CPUOffloadPolicy (ref model in GRPO/GSPO);
# it is applied explicitly from Qwen35MoeThinkerAdapter.configure_model, not
# implicitly at import time.
from .qwen3_5_moe_vision import apply_qwen3_5_vision_device_fix  # noqa: F401

__all__ = ["apply_qwen3_5_vision_device_fix"]