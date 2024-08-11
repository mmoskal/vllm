import argparse
import json
import time
import warnings
from typing import Iterable, List
import requests
import multiprocessing as mp
import numpy as np
import wandb


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


def get_streaming_response(response: requests.Response, verbose: bool) -> Iterable[List[str]]:
    for chunk in response.iter_lines(
        chunk_size=8192,
        decode_unicode=False,
        delimiter=b"\n\n",
    ):
        if chunk:
            chunk_decoded = chunk.decode("utf-8")
            items = chunk_decoded.split("\n\n")
            for item in items:
                if not item:
                    continue
                item = item[6:]  # Remove the "data: " prefix
                if item[:len('[DONE]')] == '[DONE]':
                    break
                try:
                    data = json.loads(item)
                except Exception as e:
                    if verbose:
                        print(f"Error parsing JSON ({item})")
                    continue
                output = data["choices"]
                yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = [data["choices"][i]['text'] for i in range(len(data["choices"]))]
    return output


def generate_prompt(prompt_len: int) -> str:
    tokens = ["San", "Francisco", "is", "a", "You", "are", "a", "I", "am", "a", "The", "quick", "brown", "fox", "jumps",
              "over", "the", "lazy", "dog", "random",
              "helpful", "kind", "smart", "funny", "interesting", "boring", "lazy", "active", "tall", "short", "happy",
              "sad", "angry", "excited", "calm", "friendly", ]
    # randomly chosen tokens
    import random
    prompt = " ".join(random.choices(tokens, k=prompt_len))
    return prompt


def run_request(args, result_queue, starting_delay):
    sleep_time = starting_delay / 1000
    time.sleep(sleep_time)

    model = args.model
    prompt_len = args.prompt_len
    prompt = generate_prompt(prompt_len)
    api_url = f"http://{args.host}:{args.port}/v1/completions"
    n = args.n
    stream = True
    max_tokens = args.max_tokens
    use_constraint = args.use_constraint
    verbose = args.verbose

    if stream and n > 1:
        warnings.warn("Streaming is not supported for n > 1. Setting n = 1.")
        n = 1

    if verbose:
        print(f"Prompt: {prompt!r}\n", flush=True)

    start_ts = time.time()
    token_ts = []
    context = []
    generated_tokens = 0

    for i in range(max_tokens):
        new_prompt = prompt + "".join(context)
        if verbose:
            print(f"Prompt: {new_prompt!r}\n", flush=True)
        response = post_http_request(model, new_prompt, api_url, n, stream, max_tokens)
        gen = get_streaming_response(response, verbose)
        this_time = time.time()
        if i > 0:
            token_ts.append(this_time - start_ts)
        start_ts = this_time
        a = next(gen)
        text = a[0]['text']
        if verbose:
            print(text)
        if use_constraint:
            context.append(" random")
        else:
            context.append(text)

    avg = sum(token_ts) / len(token_ts)
    if verbose:
        print(f"Average inter-token latency: {avg :.2f}s")

    # Store the result in a multiprocessing-safe queue
    result_queue.put(avg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--model", type=str, default="microsoft/Orca-2-13b")
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--use_constraint", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--poisson_delay", type=float, default=0,
                        help="Average delay (ms) of Poisson distribution (poisson_rate = 1 / poisson_delay).")
    parser.add_argument("--wandb", action="store_true", help="Log data to wandb.")
    parser.add_argument("--wandb_debug", action="store_true", help="Debug run for wandb.")
    args = parser.parse_args()

    # TODO: Query the server for some server-setup information. For example:
    #   - prefix cache enabled
    #   - sliding window enabled

    # Initialize wandb
    if args.wandb:
        wandb.init(
            project="aici",
            group="create_delete_multi",
            name=f"[{args.model}]-p-{args.processes}-t-{args.max_tokens}-c-{args.use_constraint}-pd-{args.poisson_delay}",
            tags=["create_delete_multi", "microbenchmark"] + (["run-debug"] if args.wandb_debug else ["run-standard"]),
            config={
                'prompt_len': args.prompt_len,
                'model': args.model,
                'max_tokens': args.max_tokens,
                'use_constraint': args.use_constraint,
                'processes': args.processes,
                'poisson_delay': args.poisson_delay
            })

    # Create a multiprocessing Queue to hold results
    result_queue = mp.Queue()

    # Create a Poisson
    starting_delays = [0] * args.processes
    avg_delay = args.poisson_delay
    if avg_delay != 0:
        starting_delays = np.random.poisson(avg_delay, args.processes)
        starting_delays = np.cumsum(starting_delays)

    processes = []
    for _, starting_delay in zip(range(args.processes), starting_delays):
        p = mp.Process(target=run_request, args=(args, result_queue, starting_delay))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Gather results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    if results:
        overall_avg = sum(results) / len(results)
        p50 = np.percentile(results, 50)
        p90 = np.percentile(results, 90)
        p99 = np.percentile(results, 99)

        print(f"Overall average inter-token latency across all processes: {overall_avg :.2f}s")
        print(f"P50 latency: {p50 :.2f}s")
        print(f"P90 latency: {p90 :.2f}s")
        print(f"P99 latency: {p99 :.2f}s")

        # Log data to wandb
        if args.wandb:
            wandb.log({
                "overall_avg_latency": overall_avg,
                "p50_latency": p50,
                "p90_latency": p90,
                "p99_latency": p99
            })

            result_table = wandb.Table(columns=['pid', 'latency'],
                                       data=[[i, latency] for i, latency in enumerate(results)])
            wandb.log({"per_proc_latency": result_table})

    # Finish wandb session
    wandb.finish()


if __name__ == "__main__":
    main()
