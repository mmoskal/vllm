"""Example Python client for vllm.entrypoints.api_server"""

import argparse
import json
from typing import Iterable, List

import requests


def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)


def post_http_request(prompt: str,
                      api_url: str,
                      n: int = 1,
                      max_tokens: int = 16,
                      stream: bool = False) -> requests.Response:
    headers = {
        # "User-Agent": "Test Client", 
        "Content-Type": "application/json",
    }
    pload = {
        "prompt": prompt,
        "n": n,
        # "use_beam_search": True,
        # "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
        "model": "microsoft/Orca-2-13b",
    }
    print(f"POST {api_url!r} with payload {pload!r}", flush=True)   
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        print(chunk)
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            for d in data:
                output = d["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["choices"]
    output = [d['text'] for d in output]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()
    prompt = args.prompt
    # api_url = f"http://{args.host}:{args.port}/generate"
    api_url = f"http://{args.host}:{args.port}/v1/completions"
    n = args.n
    stream = args.stream
    max_tokens = args.max_tokens

    print(f"Prompt: {prompt!r}\n", flush=True)
    response = post_http_request(
        prompt=prompt, api_url=api_url, 
        n=n, max_tokens=max_tokens, 
        stream=stream
    )

    if stream:
        num_printed_lines = 0
        for h in get_streaming_response(response):
            clear_line(num_printed_lines)
            num_printed_lines = 0
            for i, line in enumerate(h):
                num_printed_lines += 1
                print(f"Beam candidate {i}: {line!r}", flush=True)
    else:
        output = get_response(response)
        for i, line in enumerate(output):
            print(f"Beam candidate {i}: {line!r}", flush=True)
