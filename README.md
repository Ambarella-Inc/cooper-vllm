
<h1 align="center">Enable vLLM on Ambarella DevKit</h1>

# 📝Introduction
This document describes how to enable and run **vision large language model (vLLM)** on the **Ambarella DevKit** platform.

The integration is based on the **central processing unit (CPU) backend of vLLM** combined with Ambarella’s customized `transformers_amba_ext` and `transformers_amba_ext_vlm` implementations. With this setup, users can:

- Serve an **OpenAI-compatible application programming interface (API)** with **LLM**, **embedding**, **vision-language (VLM)**, and **audio** models on Ambarella DevKit
- Run inference using **Python-based workflows** without graphics processing unit (GPU) dependencies
- Reduce CPU memory pressure through **custom KV-cache optimizations** for Ambarella systems on chip (SoCs)

This guide targets developers who want to evaluate or deploy large models on Ambarella DevKit boards using vLLM while keeping compatibility with existing OpenAI-style clients.

---

# 📦 Installation
1. Create a Python environment.
vLLM currently supports **Python 3.9 to 3.12**. Ambarella recommends using `python3` or `uv` to create an isolated environment.
    ```bash
    # Use python3
    (board)# python3 -m venv venv_vllm
    (board)# source venv_vllm/bin/activate
    ```
    or
    ```bash
    # Use uv
    (board)# uv venv venv_vllm --python 3.12 --seed
    (board)# source venv_vllm/bin/activate
    ```

2. Install vLLM (CPU backend).
    Before installing vLLM, read the official vLLM CPU [installation guide](
https://docs.vllm.ai/en/v0.9.2/getting_started/installation/cpu.html) then install the required system and Python dependencies:
    ```bash
    (board)# cd $VLLM_PROJECT

    (board)# pip3 install --upgrade pip
    (board)# pip3 install "cmake>=3.26" wheel packaging ninja "setuptools-scm>=8" numpy
    (board)# pip3 install filelock==3.16.1 setuptools==77.0.3 transformers==4.52.4 huggingface-hub==0.36.0
    (board)# pip3 install dist/transformers_amba_ext-<version>.<date>-py3-none-any.whl
    (board)# pip3 install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

    (board)# SETUPTOOLS_SCM_PRETEND_VERSION=0.9.2 VLLM_TARGET_DEVICE=cpu python setup.py install
    (board)# vllm --version
    0.9.2.dev0+gb6553be1b.d20250820
    ```
    **Notes:**
    - Users can download the `transformers_amba_ext` source code from the [Ambarella GitHub](https://github.com/Ambarella-Inc/transformers_amba_ext) page and then compile the wheel package based on the `README.md` in the package.
    - Some advantages in public vLLM, such as `page-attention` and `FlashAttention`, are not supported in `cooper_vllm`.

---

# 🔥Quick Start
## Run vLLM with the DeepSeek-R1-Distill-Qwen-7B Model
1. Download the model artifacts and additional configuration files.
    Users must download the model artifacts from [Ambarella's Deepseek-R1-Distill-Qwen](https://huggingface.com/Ambarella/Deepseek-R1-Distill-Qwen) and additional configuration files listed below from [DeepSeek's DeepSeek-R1-Distill-Qwen-7B](https://huggingface.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) on Hugging Face to the model directory.
    ```bash
    (board)# ls <model_path_deepseek_7B>
      config.json
      tokenizer_config.json
      generation_config.json
    ```
2. Run vLLM as an OpenAI-compatible server.
    ```bash
    ## Optional: vLLM_0.9.2 supports the V0 architecture on the CPU
    (board)# export VLLM_CONFIG_ROOT=/tmp/
    (board)# export VLLM_CPU_OMP_THREADS_BIND="all"
    (board)# export VLLM_USE_V1=0
    (board)# export VLLM_NO_NUMA=1

    (board)# cd $WORKSPACE
    (board)# vllm serve <model_path_deepseek_7B>/ \
      --served-model-name deepseek_7B                                   \
      --host 127.0.0.1                                                  \
      --port 8000                                                       \
      --api-key EMPTY                                                   \
      --chat-template vllm/examples/tool_chat_template_deepseekr1.jinja \
      --max_model_len 32768                                             \
      --model-impl transformers_amba_ext                                \
      --load-format dummy                                               \
      --swap-space 1
    ```

3. Chat with vLLM.
    ```bash
    (board)# python3 vllm/examples/online_serving/openai_chat_completion_client.py
    ```

---

## Run vLLM with the LLaVA-OneVision-Qwen2-7B-OV-Chat Model
1. Download the model artifacts and additional configuration files.
    Users must download the model artifacts from [Ambarella's LLaVA-OneVision](https://huggingface.com/Ambarella/LLaVA-OneVision) and additional configuration files listed below for the LLM part from [LMMs-Lab's LLaVA-OneVision](https://huggingface.com/lmms-lab/llava-onevision-qwen2-7b-ov-chat) on Hugging Face to the model directory.
    ```bash
    (board)# ls <model_path_llava_onevision>
      added_tokens.json
      config.json
      generation_config.json
      special_tokens_map.json
      tokenizer_config.json
    ```

    Users must also download additional configuration files for the vision transformer (ViT) part from [Google's SigLIP](https://huggingface.com/google/siglip-so400m-patch14-384) on Hugging Face to the model directory.
    ```bash
    (board)# ls <model_path_llava_onevision>
      preprocessor_config.json
    ```

    Then update the `architectures` and `model_type` fields in `config.json`:
    ```bash
    (board)# vi <model_path_llava_onevision>/config.json
      "architectures": [
        "LlavaQwenForCausalLM"
      ],
      "model_type": "llava_qwen",
    ```
    Change it to:
    ```bash
    (board)# vi <model_path_llava_onevision>/config.json
      "architectures": [
        "LlavaOnevisionForConditionalGeneration"
      ],
      "model_type": "llava_onevision",
    ```

2. Run vLLM as an OpenAI-compatible server for the `llava_onevision` model in multi-image mode.
    ```bash
    ## Optional: vLLM_0.9.2 supports the V0 architecture on CPU
    (board)# export VLLM_CONFIG_ROOT=/tmp/
    (board)# export VLLM_CPU_OMP_THREADS_BIND="all"
    (board)# export VLLM_USE_V1=0
    (board)# export VLLM_NO_NUMA=1

    (board)# cd $WORKSPACE
    (board)# vllm serve <model_path_llava_onevision> \
      --served-model-name llava_onevision    \
      --host 127.0.0.1                       \
      --port 8000                            \
      --api-key EMPTY                        \
      --model-impl transformers_amba_ext_vlm \
      --load-format dummy                    \
      --limit-mm-per-prompt '{"image": 8}'   \
      --swap-space 1
    ```
3. Run vLLM as an OpenAI-compatible server for the `llava_onevision` model in single-image mode at high resolution (1280x720).
    Update `<model_path>/preprocessor_config.json`:
    ```json
    "size": {
      "height": 720,
      "width": 1280
    }
    ```

    Then run:
    ```bash
    ## Optional: vLLM_0.9.2 supports the V0 architecture on CPU
    (board)# export VLLM_CONFIG_ROOT=/tmp/
    (board)# export VLLM_CPU_OMP_THREADS_BIND="all"
    (board)# export VLLM_USE_V1=0
    (board)# export VLLM_NO_NUMA=1

    (board)# cd $WORKSPACE
    (board)# vllm serve <model_path_llava_onevision> \
      --served-model-name llava_onevision    \
      --host 127.0.0.1                       \
      --port 8000                            \
      --api-key EMPTY                        \
      --model-impl transformers_amba_ext_vlm \
      --load-format dummy                    \
      --hf-overrides '{"vit_mode": 0}'       \
      --swap-space 1
    ```
4. Chat with vLLM.
    ```bash
    (board)# python3 vllm/examples/online_serving/openai_chat_completion_client_for_vlm.py \
      --model_name llava_onevision \
      --input image.jpg     \
      --prompt "what can you see from this pic?" \
      --stream
    ```
---
## Run vLLM with the InternVL3_5-2B-Instruct Model
1. Download the model artifacts and additional configuration files.
    Users must download the model artifacts from [Ambarella's InternVL3.5](https://huggingface.com/Ambarella/InternVL3.5) and additional configuration files listed below from [OpenGVLab's InternVL3_5-2B](https://huggingface.com/OpenGVLab/InternVL3_5-2B) on Hugging Face to the model directory.
    ```bash
    (board)# ls <model_path_InternVL3_5>
      added_tokens.json
      config.json
      configuration_intern_vit.py
      configuration_internvl_chat.py
      conversation.py
      generation_config.json
      modeling_intern_vit.py
      modeling_internvl_chat.py
      preprocessor_config.json
      special_tokens_map.json
      tokenizer_config.json
      vocab.json
    ```
    Then add a new `preprocessor_config.json` with the configurations below to the model directory.
    ```bash
    {
      "crop_size": null,
      "crop_to_patches": false,
      "data_format": "channels_first",
      "default_to_square": true,
      "device": null,
      "do_center_crop": null,
      "do_convert_rgb": true,
      "do_normalize": true,
      "do_rescale": true,
      "do_resize": true,
      "image_mean": [
        0.485,
        0.456,
        0.406
      ],
      "image_processor_type": "GotOcr2ImageProcessorFast",
      "image_std": [
        0.229,
        0.224,
        0.225
      ],
      "input_data_format": null,
      "max_patches": 12,
      "min_patches": 1,
      "processor_class": "InternVLProcessor",
      "resample": 3,
      "rescale_factor": 0.00392156862745098,
      "return_tensors": null,
      "size": {
        "height": 448,
        "width": 448
      }
    }
    ```
2. Run vLLM as an OpenAI-compatible server for the `InternVL3` model in image mode.
    ```bash
    ## Optional: vLLM_0.9.2 supports the V0 architecture on CPU
    (board)# export VLLM_CONFIG_ROOT=/tmp/
    (board)# export VLLM_CPU_OMP_THREADS_BIND="all"
    (board)# export VLLM_USE_V1=0
    (board)# export VLLM_NO_NUMA=1

    (board)# cd $WORKSPACE
    ## Users must check the name of vision model in <model_path> and set it with an environment variable "AMBA_EXT_VLM_VIT_NAME"
    (board)# export AMBA_EXT_VLM_VIT_NAME="vision_model.bin"
    (board)# vllm serve <model_path_InternVL3_5> \
      --served-model-name internvl3_5           \
      --host 127.0.0.1                       \
      --port 8000                            \
      --api-key EMPTY                        \
      --model-impl transformers_amba_ext_vlm \
      --load-format dummy                    \
      --swap-space 0.5                       \
      --trust-remote-code
    ```
3. Chat with vLLM.
    ```bash
    (board)# python3 vllm/examples/online_serving/openai_chat_completion_client_for_vlm.py \
      --model_name internvl3_5 \
      --input image.jpg     \
      --prompt "what can you see from this pic?" \
      --stream
    ```
---

## Run vLLM with the SmolVLM2 Model
1. Download the model artifacts and additional configuration files.
    Users must download the model artifacts from [Ambarella's SmolVLM2](https://huggingface.com/Ambarella/SmolVLM2) and additional configuration files listed below from [HuggingFaceTB's SmolVLM2-500M-Video](https://huggingface.com/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) on Hugging Face to the model directory.
    ```bash
    (board)# ls <model_path_SmolVLM2>
      added_tokens.json
      chat_template.json
      config.json
      generation_config.json
      preprocessor_config.json
      processor_config.json
      special_tokens_map.json
      tokenizer_config.json
      tokenizer.json
      vocab.json
    ```
    Then update the `size` field in `preprocessor_config.json` as a unified pre-processor is used for all VLM models.
    ```
    "size": {
          "longest_edge": 2048,
    }
    ```
    Change to:
    ```
    "size": {
          "height": 512,
          "width": 512
    }
    ```
2. Run vLLM as an OpenAI-compatible server for the `SmolVLM2` model in image mode.
    ```bash
    ## Optional: vLLM_0.9.2 supports the V0 architecture on CPU
    (board)# export VLLM_CONFIG_ROOT=/tmp/
    (board)# export VLLM_CPU_OMP_THREADS_BIND="all"
    (board)# export VLLM_USE_V1=0
    (board)# export VLLM_NO_NUMA=1

    (board)# cd $WORKSPACE
    ## Users must check the name of vision model in <model_path> and set it with an environment variable "AMBA_EXT_VLM_VIT_NAME"
    (board)# export AMBA_EXT_VLM_VIT_NAME="vision_model.bin"
    (board)# vllm serve <model_path_SmolVLM2> \
      --served-model-name SmolVLM2           \
      --host 127.0.0.1                       \
      --port 8000                            \
      --api-key EMPTY                        \
      --model-impl transformers_amba_ext_vlm \
      --load-format dummy                    \
      --hf-overrides '{"vit_mode": 0}'       \
      --swap-space 0.5
    ```
3. Chat with vLLM.
    ```bash
    (board)# python3 vllm/examples/online_serving/openai_chat_completion_client_for_vlm.py \
      --model_name SmolVLM2 \
      --input image.jpg     \
      --prompt "what can you see from this pic?" \
      --stream
    ```
---

# 🐞Quality Assurance (QA)
1. vLLM v0.9.2 will install the latest version with its dependency packages, which will cause some unknown issues. Users can reinstall the lower version for packages, such as `prometheus-fastapi-instrumentator` and `mistral-common`. For the suitable version, users can choose the lowest version from the vLLM requirements text.

2. How to enable vLLM with remote procedure call (RPC) mode for the shepherd library?
    Users must add `device_ip` and `device_port` for `--hf-overrides`, as shown below. Refer to the `README.md` in `transformers_amba_ext` for more information.
    ```
    --hf-overrides '{"device_ip": "${BOARD_IP}", "device_port":8890}'
    ```

3. How to call a remote Ambarella X7 through libcrt?
    Pass `backend` and `device_id` in the same `--hf-overrides` object. `backend` must be `libcrt`. `device_id` is the remote X7 index. `transformers_amba_ext` and `transformers_amba_ext_vlm` both forward these two fields.
    ```
    --hf-overrides '{"backend": "libcrt", "device_id": 0}'
    ```

# 📜 History
- **2025-07-01**: Added vLLM with LLM model support for Ambarella SoCs
- **2025-09-01**: Added vLLM with embedding model (`Qwen3-Embedding-4B`) support for Ambarella SoCs
- **2025-12-01**: Added vLLM with VLM model (`llava-onevision`) support for Ambarella SoCs
- **2025-12-05**: Optimized KV cache management to reduce CPU memory usage on Ambarella platforms
- **2026-01-20**: Added vLLM with VLM model (`InternVL3_5-2B`) support for Ambarella SoCs
- **2026-06-25**: Added vLLM with VLM model (`Qwen3-VL-2B`) support for Ambarella SoCs
- **2026-07-23**: Added vLLM with VLM model (`Qwen3_5-VL-2B`) support for Ambarella SoCs
- **2026-08-14**: Added vLLM with VLM model (`SmolVLM2`) support for Ambarella SoCs
- **2026-10-26**: Added vLLM with LFM2.5 VL (`Lfm2Vl`) and LFM2.5 audio (`Lfm2Audio`) support for Ambarella SoCs

