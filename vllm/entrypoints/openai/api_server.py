import asyncio
import importlib
import inspect
import json
import random
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, Optional, Set
import typing

import fastapi
import pyaici
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm
import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest, ErrorResponse,
                                              RunRequest, SetTagsRequest)
from vllm.entrypoints.openai.serving_aici import AiciRunnerCompletion
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
pyaici_runner_completion: AiciRunnerCompletion

logger = init_logger(__name__)

_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = make_arg_parser()
    parser = pyaici.add_cli_args(parser)
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


def _no_aici():
    return JSONResponse({"error": "AICI runtime is not enabled"},
                        status_code=501)


@app.post("/v1/controllers")
async def upload_aici_module(request: Request):
    if not pyaici_runner_completion:
        return _no_aici()
    contents = await request.body()
    return JSONResponse(
        await
        pyaici_runner_completion.aici_runner.upload_module_async(contents))


@app.post("/v1/run")
async def aici_run(request: RunRequest, raw_request: Request):
    if not pyaici_runner_completion:
        return _no_aici()
    request_id, inst_res = \
        await pyaici_runner_completion.prep_completion(request)
    generator = pyaici_runner_completion.create_completion(
        request_id, inst_res, request, raw_request)
    return StreamingResponse(content=generator, media_type="text/event-stream")


@app.post("/v1/controllers/tags")
async def aici_set_tags(request: SetTagsRequest):
    if not pyaici_runner_completion:
        return _no_aici()
    # non-admin users can only set tags that start with their username
    auto_info = {"user": "vllm", "is_admin": True}
    r = await pyaici_runner_completion.aici_runner.set_tags(
        request.module_id, request.tags, auth_info=auto_info)
    return JSONResponse(r)


@app.get("/v1/controllers/tags")
async def aici_get_tags():
    if not pyaici_runner_completion:
        return _no_aici()
    r = await pyaici_runner_completion.aici_runner.get_tags()
    return JSONResponse(r)



# TODO(refactor): move this to a separate file
# TODO(refactor): make engine separate of this control path, ideally through a hook

class Session:
    def __init__(self, session_id: str, engine: 'AsyncLLMEngine'):
        self.session_id = session_id
        # TODO: should this hold a SequenceGroup object or a Sequence object?
        self._name2seq: Dict[str, Sequence] = {} # name -> Sequence object
        self.engine: 'AsyncLLMEngine' = engine
        pass

    async def create_prefix(self, name: str, prefix: str, following: str, sampling_params: 'SamplingParams', **kwargs):
        # Submit a sequence to run, byt at the same time fork the sequence to maintain the ref count.
        # TODO: Normally we should call the OpenAI API logic (completion, chat completion) logic to do the request
        #   We are just hacking out here.
        engine = self.engine
        # TODO: Add logic to fork from a sequence.
        request_id = random_uuid()

        # seq_id, prompt, prompt_token_ids, block_size,
        # eos_token_id, lora_request, **kwargs


        # def fork_from_existing_seq_handler(*args, following: str=None, prefix: str=None, **kwargs):
        def fork_from_existing_seq_handler(seq_id, prompt, prompt_token_ids, block_size, 
                                           eos_token_id, lora_request, **kwargs):
            """Fork a sequence from an existing sequence"""
            # Get the existing sequence from the group
            from vllm.sequence import SequenceStatus, SequenceStage, SequenceData
            following = kwargs.get("following", "")
            if following:
                seq = self._name2seq[following]  
                # TODO: Probably abstract another function `Sequence.fork_extend()`              
                new_seq = seq.fork(seq_id)
                # TODO: (1) block manager append slot with backtrack = 1, and then 
                # (2) revert the tokens inside the Sequence object 
                # Sequence.backtrack()
                # new seq need to backtrack the memory to only the prompt tokens?

                new_seq.prompt += prompt
                new_seq.tokens = new_seq.tokens[:new_seq.prefix_offset]
                new_seq.tokens += prompt_token_ids
                old_prompt_offset = new_seq.prefix_offset
                # Clear the blocks after the offset
                
                new_seq.output_text = "" # Should we trim all output?
                new_seq.output_logprobs = []
                new_seq.status = SequenceStatus.WAITING
                new_seq.stop_reason = None 
                new_seq._stage = SequenceStage.PREFILL
                new_seq.data = SequenceData(new_seq.prompt_token_ids)
                new_seq.data._num_computed_tokens = seq.prompt_offset
                new_seq.data.output_token_ids = []
                new_seq.read_offset = new_seq.prompt_offset
                
                # TODO: Check new_seq does not have garbage blocks included.
                #   Last token should not have a block in the block space.
                return new_seq
            else:
                seq = Sequence(seq_id, prompt, prompt_token_ids, block_size, 
                               eos_token_id, lora_request, **kwargs)
                return seq
            
        def register_seq_handler(seq: Sequence, *args, **kwargs):
            """Register a sequence to the session (when it finishes prefill)"""
            print("Registering sequence")
            if typing.TYPE_CHECKING:
                from vllm.core.scheduler import Scheduler
            scheduler: 'Scheduler' = kwargs['scheduler']
            new_seq_id = random.randint(1, 1000000) + 1000000 # TODO: FIXME
            new_seq = seq.fork(new_seq_id)
            scheduler.fork_seq(seq, new_seq) # where the sequence count actually increase
            self._name2seq[name] = new_seq # now the sequence and its memory is under our control
            pass

        it = engine.generate(prefix, sampling_params, request_id, 
                             following=following,
                             fork_from_existing_seq_handler=fork_from_existing_seq_handler, 
                             register_seq_handler=register_seq_handler,
                             **kwargs)
        # async iterate over it
        async for out in it:
            print(out)
        
        pass

    def delete_prefix(self, name: str):
        if name not in self._name2seq:
            raise KeyError(f"Sequence {name} not found")
        seq = self._name2seq.pop(name)
        # TODO: Free the sequence
        # seq.free() # likely, but could be more complicated than that.
        pass

    

class SessionManager():
    def __init__(self):
        self.sessions = {}
        self.engine = None

    def setup(self, engine):
        self.engine = engine

    def create_session(self):
        import uuid 
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, engine=self.engine)
        self.sessions[session_id] = session
        return session
    
    def delete_session(self, session_id: str):
        pass# TODO: Release the sequences, hence the ref count

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)
    


session_manager = SessionManager()


@app.websocket("/v1/session")
async def session_endpoint(websocket: fastapi.WebSocket):
    await websocket.accept()
    session = session_manager.create_session()
    try:
        while True:
            # TODO: Make this truly async?
            _data = await websocket.receive_text()
            actions = json.loads(_data)
            for data in actions:
                # TODO: Make it a struct
                if data["type"] == "query":
                    gpu_kv_cache = session.engine.get_gpu_kv_cache()
                    await websocket.send_text(json.dumps({"type": "query", "status": "success", "gpu_kv_cache": gpu_kv_cache}))
                    pass
                if data["type"] == "create_prefix":
                    _sampling_params = data["sampling_params"]
                    # TODO: Can max_token be 0?
                    sampling_params = SamplingParams(1, max_tokens=1, )
                    await session.create_prefix(
                        data["name"],
                        data["prefix"],
                        data["following"],
                        sampling_params,
                        **data.get("kwargs", {})
                    )
                    # send back an echo
                    await websocket.send_text(json.dumps({"type": "create_prefix", "status": "success"}))
                    pass
                if data["type"] == "delete_prefix":
                    pass
            pass
    except fastapi.websockets.WebSocketDisconnect:
        pass
    finally:
        session_manager.delete_session(session.session_id)
    pass


async def debug_session():
    session = session_manager.create_session()
    sampling_params = SamplingParams(1, max_tokens=1, )
    await session.create_prefix(
        "myseq", "hello world something something wonderful weather isn't it today haha",
        "",
        sampling_params,)

if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info("vLLM API server version %s", vllm.__version__)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    if args.aici_rt:
        config = asyncio.run(engine.get_model_config())
        dtype = str(config.dtype).replace("torch.", "").replace("float", "f")
        pyaici_runner = pyaici.runner_from_cli(args, dtype=dtype)
        pyaici_runner.fast_api()
        assert len(served_model_names) == 1
        pyaici_runner_completion = AiciRunnerCompletion(
            pyaici_runner, engine, model_config, served_model_names[0])

    openai_serving_chat = OpenAIServingChat(engine, model_config,
                                            served_model_names,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, model_config, served_model_names, args.lora_modules)

    # TODO: Remove this after debugging
    session_manager.setup(engine)
    # asyncio.run(debug_session())

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
