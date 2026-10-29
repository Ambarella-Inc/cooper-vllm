# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo,
    Qwen2VLDummyInputsBuilder,
    Qwen2VLForConditionalGeneration,
)

## Add fake Lfm2VlForConditionalGeneration to avoid checking failures.
## LFM2-VL will be included in transformers_amba_ext_vlm on Ambarella platform.
@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class Lfm2VlForConditionalGeneration(Qwen2VLForConditionalGeneration):
    pass
