"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
import warnings
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(
    model: str,
    prompt: str,
    api_url: str,
    n: int = 1,
    stream: bool = False,
    max_tokens: int = 16,
) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "n": n,
        "use_beam_search": n > 1,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192,
        decode_unicode=False,
        delimiter=b"\n\n",
    ):
        if chunk:
            chunk_decoded = chunk.decode("utf-8")
            items = chunk_decoded.split("\n\n")
            # print(items)
            for item in items:
                if not item:
                    continue
                item = item[6:]  # Remove the "data: " prefix
                if item[:len('[DONE]')] == '[DONE]':
                    break
                try:
                    data = json.loads(item)
                except Exception as e:
                    print(f"Error parsing JSON ({item})")
                    continue
                output = data["choices"]
                usage = data["usage"]
                yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = [data["choices"][i]['text'] for i in range(len(data["choices"]))]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--model", type=str, default="microsoft/Orca-2-13b")
    parser.add_argument("--max_tokens", type=int, default=16)  # New argument
    args = parser.parse_args()

    model = args.model
    prompt = args.prompt
    api_url = f"http://{args.host}:{args.port}/v1/completions"
    n = args.n
    stream = args.stream
    max_tokens = args.max_tokens  # Capture the max_tokens argument

    if stream and n > 1:
        warnings.warn("Streaming is not supported for n > 1. Setting stream to False.")
        stream = False

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(model, prompt, api_url, n, stream, max_tokens)

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                # print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)
