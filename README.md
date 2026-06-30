
<h1 align="center">Enable vLLM on Ambarella DevKit</h1>

# 📝Introduction
This document describes how to enable and run **vision large language model (vLLM)** on the **Ambarella DevKit** platform.

The integration is based on the **CPU backend of vLLM** combined with Ambarella’s customized `transformers_amba_ext` and `transformers_amba_ext_vlm` implementations. With this setup, users can:

- Serve an **OpenAI-compatible application programming interface (API)** with **LLM**, **embedding**, and **vision-language (VLM)** models on Ambarella DevKit
- Run inference using **Python-based workflows** without graphics processing unit (GPU) dependencies
- Reduce CPU memory pressure through **custom KV-cache optimizations** for Ambarella systems on chip (SoCs)

This guide targets developers who want to evaluate or deploy large models on Ambarella DevKit boards using vLLM while keeping compatibility with existing OpenAI-style clients.

---

# 📦 Installation
- Create a Python environment.
vLLM currently supports **Python 3.9–3.12**. Ambarella recommends using **python3** or **uv** to create an isolated environment.
  ```bash
  # use python3
  (board)# python3 -m venv venv_vllm
  (board)# source venv_vllm/bin/activate
  ```
  or
  ```bash
  # use uv
  (board)# uv venv venv_vllm --python 3.12 --seed
  (board)# source venv_vllm/bin/activate
  ```

- Install vLLM (CPU backend).
  Before installing vLLM, read the official vLLM CPU [installation guide](
https://docs.vllm.ai/en/v0.9.2/getting_started/installation/cpu.html). Then install the required system and Python dependencies:
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
  Note:
  - vLLM 0.9.2 will install the latest version with its dependency packages, which will cause some unknown issues. Users can reinstall the lower version for packages, such as `prometheus-fastapi-instrumentator` and `mistral-common`. For the suitable version, users can choose the lowest version from the vLLM requirements text.

---

# 🔥Quick start
## Run vLLM with the DeepSeek-R1-Distill-Qwen-7B model
- Prepare extra configuration files.
  Users must download the following configuration files from [DeepSeek in Huggingface](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) to the model directory.
  ```bash
  (board)# ls <model_path_deepseek_7B>
    config.json
    tokenizer_config.json
    generation_config.json
  ```
- Run vLLM as an OpenAI-compatible server.
  ```bash
  ## Optional: vLLM_0.9.2 supports the V0 architecture on CPU
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

- Chat with vLLM.
  ```bash
  (board)# python3 vllm/examples/online_serving/openai_chat_completion_client.py
  ```

---

## Run vLLM with the Qwen3-Embedding-4B model
- Prepare extra configuration files.
  Users must download all missing configuration files from [Qwen3-4B in Huggingface](https://huggingface.co/Qwen/Qwen3-Embedding-4B) to the model directory.
  ```bash
  (board)# ls <model_path_Qwen3-Embedding-4B>
    1_Pooling/config.json
    config.json
    config_sentence_transformers.json
    generation_config.json
    model.safetensors.index.json
    tokenizer_config.json
    vocab.json
  ```
- Run vLLM as an OpenAI-compatible server.
  ```bash
  ## Optional: vLLM_0.9.2 supports the V0 architecture on CPU
  (board)# export VLLM_CONFIG_ROOT=/tmp/
  (board)# export VLLM_CPU_OMP_THREADS_BIND="all"
  (board)# export VLLM_USE_V1=0
  (board)# export VLLM_NO_NUMA=1

  (board)# cd $WORKSPACE
  (board)# vllm serve <model_path_Qwen3-Embedding-4B> \
    --served-model-name qwen3_4B                                            \
    --host 127.0.0.1                                                        \
    --port 8000                                                             \
    --api-key EMPTY                                                         \
    --task embed                                                            \
    --model-impl transformers_amba_ext                                      \
    --load-format dummy                                                     \
    --swap-space 1
  ```
- Run an online test.
  ```bash
  (board)# python3 vllm/examples/online_serving/openai_rag_embedding_client.py
  # [[0.7727231383323669, 0.10077989846467972], [0.03620209917426109, 0.6002222895622253]]
  ```
---

## Run vLLM with the llava-onevision-qwen2-7b-ov-chat model
- Prepare extra configuration files.
  Users must download extra configuration files for LLM part from [llava-onevision in Huggingface](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov-chat) to the model directory.
  ```bash
  (board)# ls <model_path_llava_onevision>
    added_tokens.json
    config.json
    generation_config.json
    special_tokens_map.json
    tokenizer_config.json
  ```

  Users also must download extra configuration files for ViT part from [siglip in Huggingface](https://huggingface.co/google/siglip-so400m-patch14-384/) to the model directory.
  ```bash
  (board)# ls <model_path_llava_onevision>
    preprocessor_config.json
  ```

  Then, update the `architectures` field and `model_type` field in `config.json`:
  ```bash
  (board)# vi <model_path_llava_onevision>/config.json
    "architectures": [
      "LlavaQwenForCausalLM"
    ],
    "model_type": "llava_qwen",
  ```
  Change to:
  ```bash
  (board)# vi <model_path_llava_onevision>/config.json
    "architectures": [
      "LlavaOnevisionForConditionalGeneration"
    ],
    "model_type": "llava_onevision",
  ```

- Run vLLM as an OpenAI-compatible server for llava_onevision model on multi-image mode.
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
- Run vLLM as an OpenAI-compatible server for the llava_onevision model in single image mode at high resolution (1280x720).
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
- Chat with vLLM.
  ```bash
  (board)# python3 vllm/examples/online_serving/openai_chat_completion_client_for_vlm.py \
    --model_name llava_onevision \
    --input image.jpg     \
    --prompt "what can you see from this pic?" \
    --stream
  ```
---
## Run vLLM with the InternVL3-2B-Instruct model
- Prepare extra configuration files.
  Users must download extra configuration files from [InternVL3 in Huggingface](https://huggingface.co/OpenGVLab/InternVL3-2B-Instruct) to the model directory.
  ```bash
  (board)# ls <model_path_InternVL3>
    added_tokens.json
    config.json
    generation_config.json
    preprocessor_config.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
  ```
- Run vLLM as an OpenAI-compatible server for InternVL3 model in image mode.
  ```bash
  ## Optional: vLLM_0.9.2 supports the V0 architecture on CPU
  (board)# export VLLM_CONFIG_ROOT=/tmp/
  (board)# export VLLM_CPU_OMP_THREADS_BIND="all"
  (board)# export VLLM_USE_V1=0
  (board)# export VLLM_NO_NUMA=1

  (board)# cd $WORKSPACE
  ## Users must check the name of vision model in <model_path> and set it with an environment variable "AMBA_EXT_VLM_VIT_NAME"
  (board)# export AMBA_EXT_VLM_VIT_NAME="vision_model.bin"
  (board)# vllm serve <model_path_InternVL3> \
    --served-model-name internvl3           \
    --host 127.0.0.1                       \
    --port 8000                            \
    --api-key EMPTY                        \
    --model-impl transformers_amba_ext_vlm \
    --load-format dummy                    \
    --swap-space 0.5                       \
    --trust-remote-code --chat-template-content-format openai
  ```
- Chat with vLLM.
  ```bash
  (board)# python3 vllm/examples/online_serving/openai_chat_completion_client_for_vlm.py \
    --model_name internvl3 \
    --input image.jpg     \
    --prompt "what can you see from this pic?" \
    --stream
  ```
---
## Run vLLM with the Qwen3-VL-2B-Instruct model
- Prepare extra configuration files.
  Users must download extra configuration files from [Qwen3-VL in Huggingface](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) to the model directory.
  ```bash
  (board)# ls <model_path_Qwen3_VL>
    chat_template.json
    config.json
    generation_config.json
    preprocessor_config.json
    tokenizer_config.json
    video_preprocessor_config.json
    vocab.json
  ```

  Then update the `size` field in `preprocessor_config.json`, because a unified pre-processor is used for all VLM models.
  ```
  "size": {
        "longest_edge": 16777216,
        "shortest_edge": 65536
  }
  ```
  change to:
  ```
  "size": {
        "height": 512,
        "width": 768
  }
  ```
- Run vLLM as an OpenAI-compatible server for Qwen3-VL-2B model in image mode.
  ```bash
  ## Optional: vLLM_0.9.2 supports the V0 architecture on CPU
  (board)# export VLLM_CONFIG_ROOT=/tmp/
  (board)# export VLLM_CPU_OMP_THREADS_BIND="all"
  (board)# export VLLM_USE_V1=0
  (board)# export VLLM_NO_NUMA=1

  (board)# cd $WORKSPACE
  ## Users must check the name of vision model in <model_path> and set it with an environment variable "AMBA_EXT_VLM_VIT_NAME"
  (board)# export AMBA_EXT_VLM_VIT_NAME="qwen3_vl_2b_vit.bin"
  (board)# vllm serve <model_path_Qwen3_VL> \
    --served-model-name qwen3_vl           \
    --host 127.0.0.1                       \
    --port 8000                            \
    --api-key EMPTY                        \
    --model-impl transformers_amba_ext_vlm \
    --load-format dummy                    \
    --hf-overrides '{"vit_mode": 0}'       \
    --swap-space 1
  ```
- Chat with vLLM.
  ```bash
  (board)# python3 vllm/examples/online_serving/openai_chat_completion_client_for_vlm.py \
    --model_name qwen3_vl \
    --input image.jpg     \
    --prompt "what can you see from this pic?" \
    --stream
  ```
---

# 📜 History
- **2025-07-01**: Added vLLM with LLM model support for Ambarella SoCs
- **2025-09-01**: Added vLLM with embedding model(`Qwen3-Embedding-4B`) support for Ambarella SoCs
- **2025-12-01**: Added vLLM with VLM model(`llava-onevision`) support for Ambarella SoCs
- **2025-12-05**: Optimized KV cache management to reduce CPU memory usage on Ambarella platforms
- **2026-01-20**: Added vLLM with VLM model(`InternVL3-2B`) support for Ambarella SoCs
- **2026-06-25**: Added vLLM with VLM model(`Qwen3-VL-2B`) support for Ambarella SoCs
