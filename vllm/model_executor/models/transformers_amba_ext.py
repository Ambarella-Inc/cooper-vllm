# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Ambarella team.c All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only LLaMA model compatible with VLMGen weights."""
from collections.abc import Iterable
from typing import Any, ClassVar, Literal, Optional, Union
import numpy as np
import time

import torch
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from transformers_amba_ext import AutoModelForCausalLM
from vllm.logger import init_logger

logger = init_logger(__name__)

class TransformersAmbaExtForCausalLM():
    """Ambarella external accelerator model wrapper.

    This model delegates inference to external Ambarella hardware,
    so it does not use vLLM's KV cache management.
    """
    # Mark as attention-free to skip KV cache allocation in vLLM
    # The actual KV cache is managed by the Ambarella device
    is_attention_free: ClassVar[Literal[True]] = True

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type = None):

        model_config = vllm_config.model_config
        self.config = model_config.hf_config
        logger.debug(f"{self.__module__} model_config: {vllm_config.model_config}")

        model_path = model_config.model
        device_ip = model_config.hf_overrides.get("device_ip") \
            if model_config.hf_overrides is not None else None
        device_port = model_config.hf_overrides.get("device_port") \
            if model_config.hf_overrides is not None else None
        device_id = model_config.hf_overrides.get("device_id") \
            if model_config.hf_overrides is not None else None
        is_embed_model = True \
            if model_config.task is not None and model_config.task == "embed" else None
        backend = model_config.hf_overrides.get("backend") \
            if model_config.hf_overrides is not None else None
        self.no_vllm_sample = model_config.hf_overrides.get("no_vllm_sample") \
            if model_config.hf_overrides is not None else None
        self.show_perf_time = True
        logger.info(f"model_path: {model_path}, device_ip: {device_ip}, device_port: {device_port}, "
            f"device_id: {device_id}, is_embed_model: {is_embed_model}, "
            f"no_vllm_sample: {self.no_vllm_sample}, backend: {backend}")

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path = model_path,
            device_ip = device_ip, device_port = device_port,
            device_id = device_id,
            is_embed_model = is_embed_model,
            backend = backend)

    def reset(self):
        pos = self.model.reset()
        logger.info(f"{self.__module__} RESET, cur pos: {pos}")

    def run_eos(self):
        pos = self.model.run_eos()
        logger.info(f"{self.__module__} RUN_EOS, cur pos: {pos}")

    def _generate_token(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        token, pos = self.model.generate_id(input_ids)
        return token, pos

    def generate_token(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        ids_num = input_ids.shape[-1]
        time_start = time.time()
        token, pos = self._generate_token(input_ids, positions,
            intermediate_tensors, inputs_embeds, **kwargs)
        time_end = time.time()

        if ids_num > 1:
            self.show_perf_time = True
            logger.info(f"[generate_token] API prefill performance: "
                f"{ids_num / (time_end - time_start)} tokens/s at pos: {pos}")
        elif ids_num == 1 and self.show_perf_time == True:
            self.show_perf_time = False
            logger.info(f"[generate_token] API decode performance: "
                f"{ids_num / (time_end - time_start)} tokens/s at pos: {pos}")

        return token

    def _compute_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    )-> Optional[torch.Tensor]:
        input_ids_np = input_ids.numpy().astype(np.uint32).reshape(-1)
        logits, pos = self.model.generate_logits(input_ids_np)
        return torch.tensor(logits.reshape(1, -1), dtype=torch.float32), pos

    def compute_logits(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    )-> Optional[torch.Tensor]:
        ids_num = input_ids.shape[-1]
        time_start = time.time()
        logits, pos = self._compute_logits(input_ids, positions,
            intermediate_tensors, inputs_embeds, **kwargs)
        time_end = time.time()

        if ids_num > 1:
            self.show_perf_time = True
            logger.info(f"[compute_logits] API prefill performance: "
                f"{ids_num / (time_end - time_start)} tokens/s at pos: {pos}")
        elif ids_num == 1 and self.show_perf_time == True:
            self.show_perf_time = False
            logger.info(f"[compute_logits] API decode performance: "
                f"{ids_num / (time_end - time_start)} tokens/s at pos: {pos}")

        return logits

    def generate_embedding(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    )-> Optional[torch.Tensor]:
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.numpy().astype(np.uint32).reshape(-1)
        elif isinstance(input_ids, (list, tuple)):
            input_ids = np.array(input_ids, dtype=np.uint32).reshape(-1)

        hidden_states = self.model.generate_embeddings(input_ids)
        return torch.tensor(hidden_states, dtype=torch.float16)

    def state_dict(self):
        """ It's a dummy function for TransformersAmbaExtForCausalLM and
        will be invoked by DummyModelLoader.load_model
        """
        logger.info(f"{self.__module__}.state_dict Bypass")
        return {}

    def named_modules(self):
        """ It's a dummy function for TransformersAmbaExtForCausalLM and
        will be invoked by DummyModelLoader.load_model
        """
        logger.info(f"{self.__module__}.named_modules Bypass")
        return {}

    def eval(self):
        """ It's a dummy function for TransformersAmbaExtForCausalLM and
        will be invoked by DummyModelLoader.load_model
        """
        logger.info(f"{self.__module__}.eval Bypass")
        return self

    def load_weights(self,
        weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """ It's a dummy function for TransformersAmbaExtForCausalLM and
        will be invoked by DummyModelLoader.load_model
        """
        logger.info(f"{self.__module__} load_weights Bypass")
        return None