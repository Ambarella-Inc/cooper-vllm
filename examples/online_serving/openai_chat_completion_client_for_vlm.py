# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import base64
from pathlib import Path
from openai import OpenAI

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def _b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main(args):
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = args.openai_api_key
    openai_api_base = f"http://{args.host}:{args.port}/v1"
    model_name = args.model_name
    max_token_len = args.max_token_len
    temp = args.temperature
    prompt = args.prompt

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    input_fns = [Path(p) for p in args.input]
    suffixes = {p.suffix.lower() for p in input_fns}
    if len(suffixes) != 1:
        raise ValueError(f"all --input files must share one type, got {suffixes}")
    suffix = next(iter(suffixes))

    if suffix in IMAGE_EXTS:
        # LFM / many VLMs expect image before text in the user turn.
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_b64(p)}"},
            }
            for p in input_fns
        ]
        content.append({"type": "text", "text": f"{prompt}"})
        chat_completion_from_base64 = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model=model_name,
            max_completion_tokens=max_token_len,
            temperature=0.1 if temp is None else temp,
            stream=args.stream,
        )
    elif suffix in VIDEO_EXTS:
        if len(input_fns) != 1:
            raise ValueError("video mode currently supports one --input")
        chat_completion_from_base64 = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}"},
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{_b64(input_fns[0])}"
                            },
                        },
                    ],
                }
            ],
            model=model_name,
            max_completion_tokens=max_token_len,
            temperature=temp,
            stream=args.stream,
        )
    elif suffix in AUDIO_EXTS:
        # Audio before text. For multi-audio native-style prompts, put
        # <|audio_start|> where test_shepherd_vlm uses <image>, e.g.:
        #   Answer question 1 <|audio_start|> and question2 <|audio_start|> respectively
        content = [
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{_b64(p)}"},
            }
            for p in input_fns
        ]
        content.append({"type": "text", "text": f"{prompt}"})
        chat_completion_from_base64 = client.chat.completions.create(
            messages=[{"role": "user", "content": content}],
            model=model_name,
            max_completion_tokens=max_token_len,
            temperature=temp,
            stream=args.stream,
        )
    else:
        raise ValueError(f"not supported input type: {input_fns[0]}")

    print(f"[VLM]:{', '.join(str(p) for p in input_fns)}\n\n")
    if args.stream:
        for chunk in chat_completion_from_base64:
            if hasattr(chunk.choices[0].delta, "content"):
                content = chunk.choices[0].delta.content
            else:
                content = chunk
            print(content, end="", flush=True)
    else:
        result = chat_completion_from_base64.choices[0].message.content
        print(f"{result}")
    print("\n\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host IP address for vLLM server")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port number for vLLM server")
    parser.add_argument(
        "--openai_api_key", type=str, default="EMPTY", help="Port number for vLLM server")
    parser.add_argument(
        "--model_name", type=str, default="llava_onevision", help="Set model name for vLLM server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response")
    parser.add_argument(
        "--max_token_len", type=int, default=1024, help="Set max token lenght for response")
    parser.add_argument(
        "--temperature", type=float, default=None, help="Set temperature for logits sample")
    parser.add_argument(
        "--input", type=str, nargs="+", default=["image.jpg"],
        help="One or more image/video/audio paths (multi-audio: pass 2 wavs)")
    parser.add_argument(
        "--prompt", type=str, default="What's in this image?", help="Set the prompt text")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
