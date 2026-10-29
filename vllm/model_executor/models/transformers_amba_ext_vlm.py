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
"""Inference-only VLM model compatible with VLMGen weights."""
from abc import ABC
import hashlib
import numpy as np
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, Literal, Optional, TypedDict, TypeVar, Union, ClassVar, Tuple
import time
import numpy.typing as npt
import json
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import BatchEncoding, PretrainedConfig, TensorType
from transformers.image_processing_utils import get_size_dict

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs, NestedTensors)
from vllm.multimodal.parse import (ImageSize, MultiModalDataItems,
                                   MultiModalDataParser,
                                   ImageEmbeddingItems, ImageProcessorItems,
                                   VideoProcessorItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.tokenizer import AnyTokenizer

from transformers_amba_ext import AutoModelForVision, vit_mode
from vllm.logger import init_logger

logger = init_logger(__name__)

class llava_ov_special_token():
    image_token = "<image>"
    video_token = "<video>"
    image_placeholder = "<image>"
    video_placeholder = "<video>"
    shepd_image_mode_token = "<image>"
    shepd_video_mode_token = "<image>"

class internvl3_special_token():
    image_token = "<IMG_CONTEXT>"
    video_token = "<video_pad>"
    image_placeholder = "<image>"
    video_placeholder = "<video>"
    shepd_image_mode_token = "<image>"
    shepd_video_mode_token = "<video>"

class qwen3_vl_special_token():
    image_token = "<|image_pad|>"
    video_token = "<|video_pad|>"
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
    shepd_image_mode_token = "<|image_pad|>"
    shepd_video_mode_token = "<|video_pad|>"

class lfm2_vl_special_token():
    # Matches LFM2-VL HF chat_template.jinja + Ambarella pack tokenizer.
    # PromptReplacement target must equal the text that ends up in the prompt.
    image_token = "<image>"
    video_token = "<video>"
    image_placeholder = "<image>"
    video_placeholder = "<video>"
    shepd_image_mode_token = "<image>"
    shepd_video_mode_token = "<video>"

class audio_special_token():
    # Host prompt marker. Shepherd's audio slot is the same "<image>" marker
    # used for image mode; later audio packs are expected to follow this.
    audio_token = "<|audio_start|>"
    audio_placeholder = "<|audio_start|>"
    shepd_audio_mode_token = "<image>"

def _is_lfm2_audio(architectures):
    # Both packs use model_type lfm2. config.json architectures distinguishes them:
    #   Lfm2AudioForConditionalGeneration vs Lfm2VlForConditionalGeneration.
    return any("lfm2" in arch.lower() and "audio" in arch.lower()
               for arch in (architectures or []))

def _is_lfm2_vl(architectures):
    if _is_lfm2_audio(architectures):
        return False
    return any("lfm2" in arch.lower() and "vl" in arch.lower()
               for arch in (architectures or []))

def vlm_get_image_token(model_type, architectures):
    if "internvl" in model_type.lower() or \
        any("internvl" in arch.lower() for arch in architectures):
        image_token = internvl3_special_token.image_token
    elif "llava_onevision" in model_type.lower() or "smolvlm" in model_type.lower():
        image_token = llava_ov_special_token.image_token
    elif "qwen3_vl" in model_type.lower() or "qwen3_5" in model_type.lower():
        image_token = qwen3_vl_special_token.image_token
    elif _is_lfm2_vl(architectures):
        image_token = lfm2_vl_special_token.image_token
    else:
        raise ValueError(f"[vlm_get_image_token] "
            f"not supported model_type: {model_type} for transformers_amba_ext_vlm")

    return image_token

def vlm_get_video_token(model_type, architectures):
    if "internvl" in model_type.lower() or \
        any("internvl" in arch.lower() for arch in architectures):
        video_token = internvl3_special_token.video_token
    elif "llava_onevision" in model_type.lower() or "smolvlm" in model_type.lower():
        video_token = llava_ov_special_token.video_token
    elif "qwen3_vl" in model_type.lower() or "qwen3_5" in model_type.lower():
        video_token = qwen3_vl_special_token.video_token
    elif _is_lfm2_vl(architectures):
        video_token = lfm2_vl_special_token.video_token
    else:
        raise ValueError(f"[vlm_get_video_token] "
            f"not supported model_type: {model_type} for transformers_amba_ext_vlm")

    return video_token

def vlm_get_audio_placeholder(architectures):
    # Hardcoded host marker. Shepherd remaps it to "<image>" in vit_process.
    if _is_lfm2_audio(architectures):
        return audio_special_token.audio_placeholder
    return None


def vlm_get_image_placeholder(model_type, architectures):
    if "internvl" in model_type.lower() or \
        any("internvl" in arch.lower() for arch in architectures):
        return internvl3_special_token.image_placeholder
    elif "llava_onevision" in model_type.lower() or "smolvlm" in model_type.lower():
        return llava_ov_special_token.image_placeholder
    elif "qwen3_vl" in model_type.lower() or "qwen3_5" in model_type.lower():
        return qwen3_vl_special_token.image_placeholder
    elif _is_lfm2_vl(architectures):
        return lfm2_vl_special_token.image_placeholder
    else:
        raise ValueError(f"[vlm_get_image_placeholder] "
            f"not supported model_type: {model_type} for transformers_amba_ext_vlm")

def vlm_get_video_placeholder(model_type, architectures):
    if "internvl" in model_type.lower() or \
        any("internvl" in arch.lower() for arch in architectures):
        return internvl3_special_token.video_placeholder
    elif "llava_onevision" in model_type.lower() or "smolvlm" in model_type.lower():
        return llava_ov_special_token.video_placeholder
    elif "qwen3_vl" in model_type.lower() or "qwen3_5" in model_type.lower():
        return qwen3_vl_special_token.video_placeholder
    elif _is_lfm2_vl(architectures):
        return lfm2_vl_special_token.video_placeholder
    else:
        raise ValueError(f"[vlm_get_video_placeholder] "
            f"not supported model_type: {model_type} for transformers_amba_ext_vlm")

def vlm_get_shepd_image_mode_token(model_type, architectures):
    if "internvl" in model_type.lower() or \
        any("internvl" in arch.lower() for arch in architectures):
        return internvl3_special_token.shepd_image_mode_token
    elif "llava_onevision" in model_type.lower() or "smolvlm" in model_type.lower():
        return llava_ov_special_token.shepd_image_mode_token
    elif "qwen3_vl" in model_type.lower() or "qwen3_5" in model_type.lower():
        return qwen3_vl_special_token.shepd_image_mode_token
    elif _is_lfm2_vl(architectures):
        return lfm2_vl_special_token.shepd_image_mode_token
    else:
        raise ValueError(f"[vlm_get_shepd_image_mode_token] "
            f"not supported model_type: {model_type} for transformers_amba_ext_vlm")

def vlm_get_shepd_video_mode_token(model_type, architectures):
    # llava-ov suppoert video mode, but shepherd use
    #   video_token = "<image>" and vit_mode=VIT_VIDEO_MODE for video input
    if "llava_onevision" in model_type.lower() or "smolvlm" in model_type.lower():
        video_token = llava_ov_special_token.shepd_video_mode_token
    elif _is_lfm2_vl(architectures):
        video_token = lfm2_vl_special_token.shepd_video_mode_token
    else:
        raise ValueError(f"[vlm_get_shepd_video_mode_token] "
            f"not supported model_type: {model_type} for transformers_amba_ext_vlm")

    return video_token

class VLMImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: torch.Tensor
    """
    Shape:
    `(batch_size * num_images * (1 + num_patches), num_channels, height, width)`
    """

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""

VLMImageInputs = Union[VLMImagePixelInputs]


class VLMVideoPixelInputs(TypedDict):
    type: Literal["pixel_values_videos"]
    pixel_values: torch.Tensor
    """
    Shape:
    `(batch_size * num_video * num_frames, num_channels, height, width)`
    """

    num_patches: torch.Tensor
    """Shape: `(batch_size * num_images)`"""

VLMVideoInputs = Union[VLMVideoPixelInputs]

def image_transform(
    image: Image.Image,
    *,
    height: int,
    width: int,
    do_convert_rgb: Optional[bool] = None,
    do_resize: Optional[bool] = None,
    do_pad: Optional[bool] = None,
    resample: Optional[int] = None,
) -> torch.Tensor:
    if do_convert_rgb:
        image = convert_image_mode(image, 'RGB')

    if do_resize:
        interp = resample if resample is not None else T.InterpolationMode.BICUBIC
        image = T.Resize((height, width), interpolation=interp)(image)

    image = torch.from_numpy(np.array(image, dtype=np.uint8)).permute(2, 0, 1) ## (H, W, C) -> (C, H, W)
    pixel_values = image.unsqueeze(0) ## (C, H, W) -> (N, C, H, W)
    return pixel_values

def image_to_pixel_values(
    image: Image.Image,
    *,
    height: int,
    width: int,
    do_convert_rgb: Optional[bool] = None,
    do_resize: Optional[bool] = None,
    do_pad: Optional[bool] = None,
    resample: Optional[int] = None,
) -> torch.Tensor:
    return image_transform(
        image,
        height=height,
        width=width,
        do_convert_rgb=do_convert_rgb,
        do_resize=do_resize,
        do_pad=do_pad,
        resample=resample)

def video_to_pixel_values(
    video: npt.NDArray,
    *,
    height: int,
    width: int,
    do_convert_rgb: Optional[bool] = None,
    do_resize: Optional[bool] = None,
    do_pad: Optional[bool] = None,
    resample: Optional[int] = None,
) -> torch.Tensor:
    frames_list = []
    for frame in video:
        image = Image.fromarray(frame, mode="RGB")
        pil_frame = image_transform(
            image,
            height=height,
            width=width,
            do_convert_rgb=do_convert_rgb,
            do_resize=do_resize,
            do_pad=do_pad,
            resample=resample)
        frames_list.append(pil_frame)

    return torch.cat(frames_list, dim=0)

class vlm_processor_config():
    def __init__(
        self,
        processor_config: Optional[Any] = None,
    ):
        processor_config = processor_config or {}
        size_cfg = processor_config.get("size", None)
        if size_cfg is None:
            self.height = None
            self.width = None
        else:
            self.height, self.width = self._normalize_size(size_cfg)

        self.do_convert_rgb = processor_config.get("do_convert_rgb", True)
        self.do_resize = processor_config.get("do_resize", True)
        self.do_pad = processor_config.get("do_pad", False)
        self.resample = processor_config.get("resample", 3)
        # logger.info(f"[processor_config]: height: {self.height}, width: {self.width}, "
        #     f"do_convert_rgb: {self.do_convert_rgb}, do_resize: {self.do_resize}, "
        #     f"do_pad: {self.do_pad}, resample: {self.resample}")

    def _normalize_size(self, size_cfg: Union[int, dict]) -> Tuple[int, int]:
        if isinstance(size_cfg, int):
            return size_cfg, size_cfg

        if isinstance(size_cfg, dict):
            if "height" in size_cfg and "width" in size_cfg:
                return int(size_cfg["height"]), int(size_cfg["width"])

            if "shortest_edge" in size_cfg:
                size = int(size_cfg["shortest_edge"])
                return size, size

        raise ValueError(f"Unsupported size config format: {size_cfg}")


class vlm_audio_processor_config():
    """Generic audio preprocessor config, parallel to vlm_processor_config.

    Audio-vs-image is decided earlier from config.json (architectures / model_type).
    This class only reads preprocessor_config.json fields, for example::

        {
          "feature_extractor_type": "VLMAudioFeatureExtractor",
          "pack_format": "lfm_wav_seqlen_fp16",
          "sampling_rate": 16000,
          "hop_length": 160,
          "mel_len": 512
        }
    """

    def __init__(
        self,
        processor_config: Optional[Any] = None,
    ):
        processor_config = processor_config or {}
        self.sampling_rate = self._require_int(processor_config, "sampling_rate")
        self.hop_length = self._require_int(processor_config, "hop_length")
        self.mel_len = self._require_int(processor_config, "mel_len")
        pack_format = processor_config.get("pack_format", None)
        if pack_format is None:
            raise ValueError(
                "audio processor config missing required key: pack_format")
        self.pack_format = str(pack_format)

        # Derived packing geometry. Optional overrides keep the config generic
        # for packs that do not follow (mel_len - 1) * hop_length.
        self.sample_num = int(
            processor_config.get(
                "sample_num", (self.mel_len - 1) * self.hop_length))
        self.max_seq_len = int(
            processor_config.get("max_seq_len", self.mel_len - 1))
        self.packed_len = int(
            processor_config.get("packed_len", self.sample_num + 1))

    @staticmethod
    def _require_int(processor_config: dict, key: str) -> int:
        if key not in processor_config:
            raise ValueError(
                f"audio processor config missing required key: {key}")
        return int(processor_config[key])


def pack_audio_encoder_input(
    audio: npt.NDArray,
    cfg: vlm_audio_processor_config,
) -> npt.NDArray:
    """Pack a mono waveform into the blob the audio encoder net expects."""
    if cfg.pack_format == "lfm_wav_seqlen_fp16":
        # little-endian fp16 [padded waveform | seq_len]
        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=-1)
        samples = samples.reshape(-1)
        orig_len = int(samples.shape[0])
        need = cfg.sample_num
        if orig_len < need:
            samples = np.concatenate(
                [samples, np.zeros(need - orig_len, dtype=np.float32)])
        elif orig_len > need:
            samples = samples[:need]

        seq_len = min(orig_len // cfg.hop_length, cfg.max_seq_len)
        packed = np.empty(cfg.packed_len, dtype=np.float16)
        packed[:cfg.sample_num] = samples.astype(np.float16)
        packed[cfg.sample_num] = np.float16(seq_len)
        return packed

    raise ValueError(f"Unsupported audio pack_format: {cfg.pack_format}")

class VLMProcessor(ABC):
    def __init__(
        self,
        config: PretrainedConfig,
        tokenizer: AnyTokenizer,
        processor_config: Any,
        *,
        video_token: Optional[str] = None,
        image_token: Optional[str] = None,
        audio_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.vlm_processor_config = vlm_processor_config(processor_config)
        # Audio model identity is decided before this from config.json.
        # Only then read the flat audio preprocessor_config.json fields.
        self.audio_processor_config = (
            vlm_audio_processor_config(processor_config)
            if audio_token is not None else None)

        self.num_image_token = 1

        # add extra video token for video processing
        self.video_token = video_token
        self.image_token = image_token
        self.audio_token = audio_token
        logger.debug(
            f"[VLMProcessor] video_token: {video_token}, "
            f"image_token: {image_token}, audio_token: {audio_token}")

    @property
    def image_token_id(self) -> int:
        return self.tokenizer.get_vocab().get(self.image_token, None) \
            if self.image_token is not None else None

    @property
    def video_token_id(self) -> Optional[int]:
        return self.tokenizer.get_vocab().get(self.video_token, None) \
            if self.video_token is not None else None

    @property
    def audio_token_id(self) -> Optional[int]:
        return self.tokenizer.get_vocab().get(self.audio_token, None) \
            if self.audio_token is not None else None

    @property
    def supports_video(self) -> bool:
        return self.video_token_id is not None

    @property
    def supports_image(self) -> bool:
        return self.image_token_id is not None

    @property
    def supports_audio(self) -> bool:
        # Audio marker may be a multi-piece string such as <image> that is not
        # a single tokenizer vocab entry. Enable by token presence, not id.
        return self.audio_token is not None

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
    ) -> list[torch.Tensor]:
        return [
            image_to_pixel_values(
                image,
                height = self.vlm_processor_config.height,
                width = self.vlm_processor_config.width,
                do_convert_rgb = self.vlm_processor_config.do_convert_rgb,
                do_resize = self.vlm_processor_config.do_resize,
                do_pad = self.vlm_processor_config.do_pad,
                resample = self.vlm_processor_config.resample
            ) for image in images
        ]

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(images)
            image_inputs: dict[str, NestedTensors] = {
                "pixel_values":
                torch.cat(pixel_values_lst),
                "image_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst]),
            }

        return text, image_inputs

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
    ) -> list[torch.Tensor]:
        return [
            video_to_pixel_values(
                video,
                height = self.vlm_processor_config.height,
                width = self.vlm_processor_config.width,
                do_convert_rgb = self.vlm_processor_config.do_convert_rgb,
                do_resize = self.vlm_processor_config.do_resize,
                do_pad = self.vlm_processor_config.do_pad,
                resample = self.vlm_processor_config.resample
            ) for video in videos
        ]

    def _preprocess_video(
        self,
        text: list[str],
        videos: list[npt.NDArray],
    ):
        if len(videos) == 0 or not self.supports_video:
            video_inputs = {}
        else:
            pixel_values_lst_video = self._videos_to_pixel_values_lst(videos)
            video_inputs: dict[str, NestedTensors] = {
                "pixel_values_video":
                torch.cat(pixel_values_lst_video),
                "video_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst_video]),
            }

        return text, video_inputs

    def _audios_to_features_lst(
        self,
        audios: list[npt.NDArray],
    ) -> list[torch.Tensor]:
        packed = [
            pack_audio_encoder_input(audio, self.audio_processor_config)
            for audio in audios
        ]
        return [torch.from_numpy(item).unsqueeze(0) for item in packed]

    def _preprocess_audio(
        self,
        text: list[str],
        audios: list[npt.NDArray],
    ):
        if len(audios) == 0 or not self.supports_audio:
            audio_inputs = {}
        else:
            features_lst = self._audios_to_features_lst(audios)
            # call_hf_processor casts every floating tensor to the model dtype
            # (bf16). fp16 -> bf16 -> fp16 does not round-trip, so the packed
            # clip bytes no longer match prepare_audio_encoder_input.py.
            # Store the fp16 payload as uint8 so that cast is skipped.
            packed_fp16 = torch.cat(features_lst, dim=0).contiguous()
            audio_inputs: dict[str, NestedTensors] = {
                "audio_features":
                packed_fp16.view(torch.uint8),
                "audio_num_clips":
                torch.tensor([len(item) for item in features_lst]),
            }

        return text, audio_inputs

    def _make_batch_input(self,
                          input_item: Optional[Union[Any, list[Any]]] = None):
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        videos: Optional[Union[npt.NDArray, list[npt.NDArray]]] = None,
        audios: Optional[Union[npt.NDArray, list[npt.NDArray]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Mapping[str, NestedTensors]:
        text, images, videos, audios = [
            self._make_batch_input(x) for x in (text, images, videos, audios)
        ]
        logger.debug(
            f"text: {text}, images: {images}, videos: {videos}, audios: {audios}")
        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
        )

        text, video_inputs = self._preprocess_video(
            text=text,
            videos=videos,
        )

        text, audio_inputs = self._preprocess_audio(
            text=text,
            audios=audios,
        )

        text_inputs = self.tokenizer(text)

        return {
            **BatchEncoding(text_inputs, tensor_type=return_tensors),
            **image_inputs,
            **video_inputs,
            **audio_inputs,
        }

class VLMProcessingInfo(BaseProcessingInfo):
    def _is_lfm2_audio_model(self) -> bool:
        hf_config = self.get_hf_config()
        return _is_lfm2_audio(hf_config.architectures)

    @property
    def supports_image(self):
        return self.get_hf_processor().supports_image

    @property
    def supports_video(self):
        return self.get_hf_processor().supports_video

    @property
    def supports_audio(self):
        return self.get_hf_processor().supports_audio

    def get_supported_mm_limits(self):
        image_limit = {"image": None} if self.supports_image else {}
        video_limit = {"video": None} if self.supports_video else {}
        audio_limit = {"audio": None} if self.supports_audio else {}
        logger.debug(
            f"image_limit: {image_limit}, video_limit: {video_limit}, "
            f"audio_limit: {audio_limit}")
        return {**image_limit, **video_limit, **audio_limit}

    def get_image_token(self) -> Optional[str]:
        if self._is_lfm2_audio_model():
            return None
        return vlm_get_image_token(
            self.get_hf_config().model_type,
            self.get_hf_config().architectures)

    def get_video_token(self) -> Optional[str]:
        if self._is_lfm2_audio_model():
            return None
        return vlm_get_video_token(
            self.get_hf_config().model_type,
            self.get_hf_config().architectures)

    def get_audio_token(self) -> Optional[str]:
        hf_config = self.get_hf_config()
        return vlm_get_audio_placeholder(hf_config.architectures)

    def _get_processor_config(self) -> dict[str, Any]:
        processor_config = self.ctx.get_hf_image_processor_config()
        if processor_config:
            return processor_config

        config_path = Path(self.ctx.model_config.model) / "preprocessor_config.json"
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                processor_config = json.load(f)
            logger.debug(
                f"[VLMProcessingInfo] loading preprocessor_config from {config_path}")
            return processor_config

        raise FileNotFoundError(
            f"preprocessor_config.json not found in model path: {config_path}. "
            "Image/video models need size/resize fields; audio models need "
            "sampling_rate, hop_length, mel_len, and pack_format.")

    def get_hf_processor(
        self,
        **kwargs: object,
    ) -> VLMProcessor:
        kwargs["video_token"] = self.get_video_token()
        kwargs["image_token"] = self.get_image_token()
        kwargs["audio_token"] = self.get_audio_token()
        logger.debug(f"[VLMProcessingInfo] get_hf_processor kwargs: {kwargs}")

        return self.ctx.init_processor(
            VLMProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            processor_config=self._get_processor_config(),
            **kwargs,
        )

class VLMDummyInputsBuilder(
        BaseDummyInputsBuilder[VLMProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)
        logger.debug(
            f"[get_dummy_text] num_images: {num_images}, "
            f"num_videos: {num_videos}, num_audios: {num_audios}")
        hf_config = self.info.get_hf_config()
        dummy = ""
        if self.info.supports_image and num_images:
            dummy += vlm_get_image_placeholder(
                hf_config.model_type, hf_config.architectures) * num_images
        if self.info.supports_video and num_videos:
            dummy += vlm_get_video_placeholder(
                hf_config.model_type, hf_config.architectures) * num_videos
        if self.info.supports_audio and num_audios:
            audio_ph = vlm_get_audio_placeholder(hf_config.architectures)
            if audio_ph is None:
                raise ValueError(
                    "audio placeholder is None; cannot build dummy text")
            dummy += audio_ph * num_audios
        return dummy

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        dummy_image = {}
        dummy_video = {}
        dummy_audio = {}

        if self.info.supports_image:
            config = self.info.get_hf_config()
            image_size: int = getattr(
                getattr(config, "vision_config", None), "image_size", 448)
            num_images = mm_counts.get("image", 0)
            dummy_image = {
                "image":
                self._get_dummy_images(width=image_size,
                                       height=image_size,
                                       num_images=num_images)
            }
            if self.info.supports_video:
                num_videos = mm_counts.get("video", 0)
                dummy_video = {
                    "video":
                    self._get_dummy_videos(width=image_size,
                                           height=image_size,
                                           num_frames=1,
                                           num_videos=num_videos)
                }

        if self.info.supports_audio:
            audio_cfg = self.info.get_hf_processor().audio_processor_config
            dummy_audio = {
                "audio":
                self._get_dummy_audios(
                    length=audio_cfg.sample_num,
                    num_audios=mm_counts.get("audio", 0))
            }

        logger.debug(
            f"[get_dummy_mm_data] dummy_image: {dummy_image}, "
            f"dummy_video: {dummy_video}, dummy_audio: {dummy_audio}")
        return {**dummy_image, **dummy_video, **dummy_audio}

class VLMMultiModalProcessor(
        BaseMultiModalProcessor[VLMProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> bool:
        # VLMProcessor only packs image/video/audio tensors and tokenizes text
        # as-is; it does not expand mm placeholders. Let vLLM apply updates
        # (with text-match fallback) so <image> / audio markers are found.
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, NestedTensors]:
        processed_outputs = super()._call_hf_processor(prompt, mm_data,
                                                       mm_kwargs)
        hf_processor = self.info.get_hf_processor(**mm_kwargs)
        logger.debug(f"processed_outputs: {processed_outputs}, "
            f"hf_processor.image_token_id: {hf_processor.image_token_id}, "
            f"hf_processor.video_token_id: {hf_processor.video_token_id}, "
            f"hf_processor.audio_token_id: {hf_processor.audio_token_id}")

        if self.info.supports_image and \
            (image_token_id := hf_processor.image_token_id) is not None:
            processed_outputs["image_token_id"] = torch.tensor(image_token_id)

        if self.info.supports_video and \
            (video_token_id := hf_processor.video_token_id) is not None:
            processed_outputs["video_token_id"] = torch.tensor(video_token_id)

        if self.info.supports_audio and \
            (audio_token_id := hf_processor.audio_token_id) is not None:
            processed_outputs["audio_token_id"] = torch.tensor(audio_token_id)

        return processed_outputs

    def _get_data_parser(self) -> MultiModalDataParser:
        if self.info.supports_audio:
            # Match LFM2.5 prepare_audio_encoder_input.py: torchaudio
            # sinc_interp_hann (not librosa) so packed clips match golden bins.
            sampling_rate = self.info.get_hf_processor().audio_processor_config.sampling_rate
            return MultiModalDataParser(
                target_sr=sampling_rate,
                audio_resample_method="torchaudio",
            )
        return MultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, NestedTensors],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        if self.info.supports_image:
            image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
            num_images = len(image_num_patches)
            image_fields = dict(
                pixel_values=MultiModalFieldConfig.flat_from_sizes("image", image_num_patches),
                image_num_patches=MultiModalFieldConfig.batched("image"),
                image_embeds=MultiModalFieldConfig.batched("image"),
                image_token_id=MultiModalFieldConfig.shared("image", num_images),
            )
        else:
            num_images = 0
            image_fields = {}

        if self.info.supports_video:
            video_num_patches = hf_inputs.get("video_num_patches", torch.empty(0))
            num_videos = len(video_num_patches)
            video_fields = dict(
                pixel_values_video=MultiModalFieldConfig.flat_from_sizes("video", video_num_patches),
                video_num_patches=MultiModalFieldConfig.batched("video"),
                video_token_id=MultiModalFieldConfig.shared("video", num_videos),
            )
        else:
            num_videos = 0
            video_fields = {}

        if self.info.supports_audio:
            audio_num_clips = hf_inputs.get("audio_num_clips", torch.empty(0))
            num_audios = len(audio_num_clips)
            audio_fields = dict(
                audio_features=MultiModalFieldConfig.flat_from_sizes(
                    "audio", audio_num_clips),
                audio_num_clips=MultiModalFieldConfig.batched("audio"),
            )
            if "audio_token_id" in hf_inputs:
                audio_fields["audio_token_id"] = MultiModalFieldConfig.shared(
                    "audio", num_audios)
        else:
            num_audios = 0
            audio_fields = {}

        logger.debug(f"[_get_mm_fields_config] "
            f"num_images: {num_images}, num_videos: {num_videos}, "
            f"num_audios: {num_audios}, "
            f"image_fields: {image_fields}, video_fields: {video_fields}, "
            f"audio_fields: {audio_fields}")

        return image_fields | video_fields | audio_fields

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        model_type = hf_config.model_type.lower()
        architectures = hf_config.architectures
        prompt_repl: list[PromptUpdate] = []

        if self.info.supports_image:
            def image_repl_fn(idx: int):
                return PromptUpdateDetails.select_text(
                    hf_processor.image_token, hf_processor.image_token)

            image_target = vlm_get_image_placeholder(hf_config.model_type, architectures)
            prompt_repl.append(
                PromptReplacement(
                    modality="image",
                    target=image_target,
                    replacement=image_repl_fn,
                )
            )

        if self.info.supports_video:
            def video_repl_fn(idx: int):
                return PromptUpdateDetails.select_text(
                    hf_processor.video_token, hf_processor.video_token)

            video_target = vlm_get_video_placeholder(hf_config.model_type, architectures)
            prompt_repl.append(
                PromptReplacement(
                    modality="video",
                    target=video_target,
                    replacement=video_repl_fn,
                )
            )

        if self.info.supports_audio:
            def audio_repl_fn(idx: int):
                return PromptUpdateDetails.select_text(
                    hf_processor.audio_token, hf_processor.audio_token)

            audio_target = vlm_get_audio_placeholder(architectures)
            if audio_target is None:
                raise ValueError(
                    "audio PromptReplacement target is None; "
                    f"model_type={hf_config.model_type}, "
                    f"architectures={architectures}")
            prompt_repl.append(
                PromptReplacement(
                    modality="audio",
                    target=audio_target,
                    replacement=audio_repl_fn,
                )
            )

        return prompt_repl


# VLMProcessingInfo covers image / video / audio. Audio packing is configured by
# preprocessor_config.json (same role as Qwen2AudioProcessingInfo + feature extractor).
@MULTIMODAL_REGISTRY.register_processor(
    VLMMultiModalProcessor,
    info=VLMProcessingInfo,
    dummy_inputs=VLMDummyInputsBuilder)
class TransformersAmbaExtForVLM():
    """Ambarella external accelerator model wrapper.

    This model delegates inference to external Ambarella hardware,
    so it does not use vLLM's KV cache management.
    """
    # Mark as attention-free to skip KV cache allocation in vLLM
    # The actual KV cache is managed by the Ambarella device
    is_attention_free: ClassVar[Literal[True]] = True
    # Required so ModelConfig accepts --limit-mm-per-prompt / OpenAI image|audio
    supports_multimodal: ClassVar[Literal[True]] = True

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 layer_type = None):

        model_config = vllm_config.model_config
        self.config = model_config.hf_config
        logger.debug(f"{self.__module__} model_config: {vllm_config.model_config}")

        model_path = model_config.model
        trust_remote_code = getattr(model_config, "trust_remote_code", None)
        overrides = model_config.hf_overrides or {}
        device_ip = overrides.get("device_ip")
        device_port = overrides.get("device_port")
        device_id = overrides.get("device_id")
        backend = overrides.get("backend")
        vit_mode = overrides.get("vit_mode")
        # 0: single image, 1: multi image, 2: video
        self.vit_mode = vit_mode if vit_mode is not None else 1
        self.no_vllm_sample = overrides.get("no_vllm_sample")
        self.show_perf_time = True

        # Same as test_shepherd_vlm --in N. Prefer explicit hf_overrides,
        # else derive from --limit-mm-per-prompt.
        max_img_num = overrides.get("max_img_num")
        if max_img_num is None:
            mm_cfg = getattr(model_config, "multimodal_config", None)
            limits = []
            if mm_cfg is not None:
                for modality in ("audio", "image", "video"):
                    try:
                        lim = mm_cfg.get_limit_per_prompt(modality)
                    except Exception:
                        lim = None
                    if lim:
                        limits.append(int(lim))
            max_img_num = max(limits) if limits else 1

        logger.info(f"model_path: {model_path}, device_ip: {device_ip}, device_port: {device_port}, "
            f"device_id: {device_id}, backend: {backend}, vit_mode: {self.vit_mode}, "
            f"trust_remote_code: {trust_remote_code}, "
            f"no_vllm_sample: {self.no_vllm_sample}, max_img_num: {max_img_num}")

        self.model = AutoModelForVision.from_pretrained(
            pretrained_model_path = model_path,
            device_ip = device_ip, device_port = device_port,
            device_id = device_id, backend = backend,
            trust_remote_code = trust_remote_code,
            max_img_num = max_img_num)

    def reset(self):
        pos = self.model.reset()
        logger.info(f"{self.__module__} RESET, cur pos: {pos}")

    def run_eos(self):
        pos = self.model.run_eos()
        logger.info(f"{self.__module__} RUN_EOS, cur pos: {pos}")

    def vit_process(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        pixel_values_image = kwargs.pop("pixel_values", None)
        pixel_values_video = kwargs.pop("pixel_values_video", None)
        audio_features = kwargs.pop("audio_features", None)

        text = self.model.decode(input_ids)
        if pixel_values_image is not None: # for single image & multi image
            pixel_values = pixel_values_image
            vit_mode_ext = self.vit_mode

            if vlm_get_image_placeholder(self.config.model_type, self.config.architectures) in text:
                repl_src = vlm_get_image_placeholder(self.config.model_type, self.config.architectures)
            elif vlm_get_image_token(self.config.model_type, self.config.architectures) in text:  ## Add Bugfix for internvl3.5-vl
                repl_src = vlm_get_image_token(self.config.model_type, self.config.architectures)
            else:
                raise ValueError(f"image placeholder or token not found in text: {text}")
            repl_dst = vlm_get_shepd_image_mode_token(
                self.config.model_type, self.config.architectures)
            pixel_values_reshape = pixel_values.view(-1, *pixel_values.shape[-3:])
            pixel = pixel_values_reshape.numpy().astype(np.uint8)
            self.model.tokenizer_image_token(pixel, vit_mode = vit_mode_ext)
            logger.info(f"pixel_values.shape: {pixel_values_reshape.shape}, "
                f"input_ids.shape: {input_ids.shape}")
        elif pixel_values_video is not None: # for video
            pixel_values = pixel_values_video
            vit_mode_ext = vit_mode.VIDEO
            repl_src = vlm_get_video_placeholder(
                self.config.model_type, self.config.architectures)
            repl_dst = vlm_get_shepd_video_mode_token(
                self.config.model_type, self.config.architectures)
            pixel_values_reshape = pixel_values.view(-1, *pixel_values.shape[-3:])
            pixel = pixel_values_reshape.numpy().astype(np.uint8)
            self.model.tokenizer_image_token(pixel, vit_mode = vit_mode_ext)
            logger.info(f"pixel_values.shape: {pixel_values_reshape.shape}, "
                f"input_ids.shape: {input_ids.shape}")
        elif audio_features is not None:
            vit_mode_ext = vit_mode.AUDIO
            # chat_template.jinja keeps <|audio_start|>. Decode, then replace
            # with shepherd's <image>, same as the image/video path below.
            if audio_special_token.audio_placeholder in text:
                repl_src = audio_special_token.audio_placeholder
            elif audio_special_token.audio_token in text:
                repl_src = audio_special_token.audio_token
            else:
                raise ValueError(
                    f"audio placeholder or token not found in text: {text!r}")
            repl_dst = audio_special_token.shepd_audio_mode_token
            # Packed clips are fp16 bytes stored as uint8 so the multimodal
            # dtype cast cannot turn them into bf16. View them back here.
            audio_t = audio_features.detach().cpu().contiguous()
            if audio_t.dtype == torch.uint8:
                audio_t = audio_t.view(torch.float16)
            elif audio_t.dtype != torch.float16:
                logger.warning(
                    "audio_features dtype %s is not preserved fp16 bytes; "
                    "casting to float16", audio_t.dtype)
                audio_t = audio_t.to(dtype=torch.float16)
            audio = np.ascontiguousarray(audio_t.numpy())
            while audio.ndim > 2:
                audio = audio.reshape(-1, audio.shape[-1])
            audio_sha = hashlib.sha256(audio.tobytes()).hexdigest()[:16]
            logger.info(
                f"audio_features raw dtype={audio_features.dtype}, "
                f"packed dtype={audio.dtype}, shape={audio.shape}, "
                f"nbytes={audio.nbytes}, sha256_16={audio_sha}")
            self.model.tokenizer_image_token(audio, vit_mode=vit_mode_ext)
        else:
            raise ValueError(
                "pixel_values, pixel_values_video, or audio_features is None from kwargs")

        text = text.replace(repl_src, repl_dst)
        merged_ids = self.model.encode(text, no_sys_prompt=True)
        logger.debug(f"repl_src: {repl_src}, repl_dst: {repl_dst}")
        logger.debug(f"self.config.model_type: {self.config.model_type}, "
                    f"self.config.architectures: {self.config.architectures}")
        logger.debug(f"text: {text}")
        logger.info(f"merged_ids.shape: {merged_ids.shape}")

        return merged_ids

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