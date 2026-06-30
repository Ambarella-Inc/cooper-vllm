# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    parser.add_argument(
        "--request_id", type=str, default=None, help="Set the request id (optional)")
    return parser.parse_args()


def main(args):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id
    stream = args.stream
    request_id = args.request_id

    messages = []
    while True:
        input_string = input("\n[User]: ")
        if input_string == "":
            continue
        if (input_string == "EOT") and (len(input_string) == 3):
            break
        messages.append({"role": "user", "content": input_string})

        # Chat Completion API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            stream=stream,
            extra_body = {"request_id": request_id} if request_id is not None else None
        )

        assistant_response = ""
        if args.stream:
            for chunk in chat_completion:
                if hasattr(chunk.choices[0].delta, "content") and \
                    chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_response += content
        else:
            assistant_response = chat_completion.choices[0].message.content
            print(assistant_response)

        messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    args = parse_args()
    main(args)
