# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioMultiModalProcessor,
    Qwen2AudioProcessingInfo,
    Qwen2AudioDummyInputsBuilder,
    Qwen2AudioForConditionalGeneration,
)

## Add fake Lfm2AudioForConditionalGeneration to avoid checking failures.
## LFM2.5 audio will be included in transformers_amba_ext_vlm on Ambarella platform.
@MULTIMODAL_REGISTRY.register_processor(Qwen2AudioMultiModalProcessor,
                                        info=Qwen2AudioProcessingInfo,
                                        dummy_inputs=Qwen2AudioDummyInputsBuilder)
class Lfm2AudioForConditionalGeneration(Qwen2AudioForConditionalGeneration):
    pass
