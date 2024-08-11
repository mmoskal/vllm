import asyncio
from argparse import Namespace

from fastapi import FastAPI

import vllm.entrypoints.openai.api_server as vllm_api_server
from vllm.engine.protocol import AsyncEngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client,
    TIMEOUT_KEEP_ALIVE,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_tokenization import OpenAIServingTokenization
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger

logger = init_logger('aici_mods.api_server')

AICI_API_VERSION = "0.0.1"


async def run_server(args, **uvicorn_kwargs) -> None:
    logger.info("vLLM-AICI API server version %s", AICI_API_VERSION)
    logger.info("args: %s", args)

    async with build_async_engine_client(args) as async_engine_client:
        app = await vllm_api_server.init_app(async_engine_client, args)

        shutdown_task = await serve_http(
            app,
            engine=async_engine_client,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    asyncio.run(run_server(args))
