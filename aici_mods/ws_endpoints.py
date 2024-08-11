"""
Websocket endpoint for sequence level controlling.
"""
from typing import List, Dict, Union

import fastapi

import vllm.entrypoints.openai.api_server as vllm_api_server
from fastapi import APIRouter, FastAPI, Request

from vllm.engine.async_llm_engine import AsyncStream
from vllm.logger import init_logger
from vllm import AsyncLLMEngine, SamplingParams, TextPrompt, TokensPrompt

logger = init_logger('aici_mods.ws_endpoints')


def get_router() -> FastAPI:
    return vllm_api_server.router


def get_engine() -> AsyncLLMEngine:
    engine = vllm_api_server.async_engine_client
    assert isinstance(engine, AsyncLLMEngine)
    return engine


router = get_router()


class Session:
    def __init__(self, websocket: fastapi.WebSocket):
        self.active_sequences = {}
        self.websocket = websocket
        pass

    async def handle_data(self):
        websocket = self.websocket

        while True:
            data = await websocket.receive_json()
            # TODO(mike): Formalize these actions / kwargs into structs.
            action = data['action']
            kwargs = data['kwargs']
            try:
                if action == 'create':
                    request_id = kwargs.pop('request_id')
                    self.create_sequence(**kwargs)
                elif action == 'delete':
                    self.delete_sequence(**kwargs)
                    pass
            except Exception as e:
                # TODO(mike): (error) handler error properly. Don't abort the session.
                pass
        pass

    def create_sequence(
        self,
        request_id: str,
        prompt_text: str = None,
        prompt_token_ids: List[int] = None,
        sampling_params: Dict = None,
    ) -> AsyncStream:
        engine: AsyncLLMEngine = get_engine()
        sampling_params = SamplingParams(
            **(sampling_params or {})
        )
        assert prompt_text or prompt_token_ids
        assert not (prompt_text and prompt_token_ids)

        prompt = None
        if prompt_text:
            prompt = TextPrompt(prompt=prompt_text)
        if prompt_token_ids:
            prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
            pass

        # TODO(mike): (patch) AsyncLLMEngine.add_request_ is a patch.
        stream: 'AsyncStream' = engine.add_request_(
            # TODO(mike): (security) should create an indirection of request_id to avoid duplication.
            request_id, prompt,
            sampling_params,
        )
        self.active_sequences[request_id] = stream
        return stream

    def delete_sequence(self, request_id: Union[int, str]):
        engine = get_engine()
        # TODO(mike): AsyncLLMEngine.abort is async. Bypass it by calling _abort directly.
        engine._abort(request_id)
        Ò
        self.active_sequences.pop(request_id, None)
        print(f"Aborted request_id: {request_id}")
        return

    pass


async def session_handler(websocket: fastapi.WebSocket):
    session = Session(websocket)

    pass


@router.websocket("/v1/session")
async def websocket_endpoint(websocket: fastapi.WebSocket):
    await websocket.accept()
    try:
        await session_handler(websocket)
        # TODO(mike): Add exception handling logics
    except fastapi.WebSocketDisconnect as e:
        logger.info("Websocket disconnected.")
        return
    except Exception as e:
        raise e

    return
