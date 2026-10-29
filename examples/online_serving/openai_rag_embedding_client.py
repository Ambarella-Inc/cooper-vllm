# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def main():
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    #an example case from [huggingface](https://huggingface.co/Qwen/Qwen3-Embedding-4B) to test Qwen3 4B
    # Each query must come with a one-sentence instruction that describes the task
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = [
        get_detailed_instruct(task, 'What is the capital of China?'),
        get_detailed_instruct(task, 'Explain gravity')
    ]
    # No need to add instruction for retrieval documents
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
    ]
    input_texts = queries + documents

    responses = client.embeddings.create(
        # ruff: noqa: E501
        input=input_texts,
        model=model,
    )

    embeddings = torch.tensor([data.embedding for data in responses.data])
    scores = (embeddings[:2] @ embeddings[2:].T)
    print(scores.tolist())
    # [[0.7727231383323669, 0.10077989846467972], [0.03620209917426109, 0.6002222895622253]]


if __name__ == "__main__":
    main()
