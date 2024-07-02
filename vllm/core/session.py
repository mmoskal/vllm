import asyncio
import random
from collections import deque
from typing import TYPE_CHECKING, Dict, Optional, List

from vllm.utils import random_uuid

if TYPE_CHECKING:
    from fastapi import WebSocket
    from vllm import LLMEngine, SamplingParams, AsyncLLMEngine
    from vllm.sequence import SequenceGroup, Sequence
    from vllm.engine.async_llm_engine import AsyncStream


class SessionManager:

    def __init__(self, engine: 'AsyncLLMEngine'):
        self.engine = engine
        self.sessions: Dict[str, 'Session'] = {}
        pass

    def create_session(self, session_id: str = None) -> 'Session':
        session_id = session_id or random_uuid()
        session = Session(self.engine, session_id)
        self.sessions[session_id] = session
        return session

    def delete_session(self):
        raise NotImplementedError
        pass

    def get_session(self, session_id: str) -> Optional['Session']:
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        return self.sessions.get(session_id)


class Session:

    def __init__(self, engine: 'AsyncLLMEngine', session_id: str):
        self.session_id: str = session_id
        self.engine: 'AsyncLLMEngine' = engine
        self.user_requests: Dict[str, UserRequest] = {}
        # UserRequest reference count.
        self.req_ref_count: Dict[str, int] = {}
        self.socket: Optional['WebSocket'] = None
        self.outgoing_streams: Dict[str, 'AsyncStream'] = {}
        self.outgoing_message: 'asyncio.Queue' = asyncio.Queue()
        pass

    @property
    def scheduler(self):
        return self.engine.engine.scheduler

    async def loop_handle_incoming_tasks(self):

        async def triage_task(action: str, kwargs: dict):
            if action == "create_sequence":
                name = kwargs['name']
                prompt = kwargs['prompt']
                prompt_token_ids = kwargs['prompt_token_ids']
                following = kwargs['following']
                _sampling_params = kwargs['sampling_params']
                sampling_params = SamplingParams(**_sampling_params)
                await self.create_sequence(
                    name,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    following=following,
                    sampling_params=sampling_params,
                )
                return
            if action == "pause_sequence":
                name = kwargs['name']
                await self.pause_sequence(name)
                return
            if action == "resume_sequence":
                name = kwargs['name']
                await self.resume_sequence(name)
                return
            if action == "stop_sequence":
                name = kwargs['name']
                await self.stop_sequence(name)
                return
            raise KeyError(f"Unknown action: {action}")

        while True:
            data = await self.socket.receive_json()
            # TODO: Triage the data to perform actions.
            action = data['action']
            kwargs = data['kwargs']
            # action in ['create_sequence', 'pause_sequence', 'resume_sequence', 'stop_sequence']
            await triage_task(action, kwargs
                              )  # everything here is on the critical path.
            await asyncio.sleep(0)
            pass
        pass

    async def loop_handle_outgoing_streams(self):
        # Query all existing streams.
        # If there are content, stream it to the other side.
        # TODO: Define a max handle stream for the iteration.
        while True:
            keys = list(self.outgoing_streams.keys())
            new_outgoing_streams = {}
            for name in keys:
                stream = self.outgoing_streams[name]
                if stream.finished:
                    continue
                # Stream the content to the user.
                try:
                    item = await anext(stream)
                    from vllm import CompletionOutput
                    # assert isinstance(item.outputs, CompletionOutput)
                    if isinstance(item.output, CompletionOutput):
                        result = dict(
                            result_type="token",
                            name=name,
                            text=item.outputs.text,
                            token_ids=item.outputs.token_ids,
                        )
                    else:
                        result = dict(
                            result_type="token",
                            name=name,
                            text="<todo>",
                            token_ids="<todo>",
                        )
                        pass
                    await self.socket.send_json(result)
                except Exception as e:
                    print(f"While looping over stream {name}, got error: {e}")
                    pass
                new_outgoing_streams[name] = stream
            self.outgoing_streams = new_outgoing_streams
            await asyncio.sleep(0)
        pass

    async def start_event_loop(self):
        # Start all event loops
        await asyncio.gather(
            self.loop_handle_incoming_tasks(),
            self.loop_handle_outgoing_streams(),
        )
        pass

    def register_websocket(self, socket):
        self.socket = socket
        return

    def _incr_ref_count(self, name: str) -> None:
        """Increment the reference count of the request and its dependent requests."""
        assert name in self.user_requests
        if name not in self.req_ref_count:
            self.req_ref_count[name] = 0

        while True:
            self.req_ref_count[name] += 1
            this_request = self.user_requests[name]
            name = this_request.following
            if not name:
                break
            assert name in self.user_requests
            assert name in self.req_ref_count
        return

    def _decr_ref_count(self, name: str) -> None:
        """Decrement the reference count of the request and its dependent requests."""
        assert name in self.user_requests
        assert name in self.req_ref_count
        while True:
            self.req_ref_count[name] -= 1
            this_request = self.user_requests[name]
            name = this_request.following
            if not name:
                break
            assert name in self.user_requests
            assert name in self.req_ref_count
        return

    async def create_sequence(
        self,
        name: str = None,
        prompt: str = None,
        prompt_token_ids: List[int] = None,
        following: str = None,
        sampling_params: SamplingParams = None,
        eager: bool = False,
        stream_to_user: bool = False,
    ):
        request_id = random_uuid()
        request = UserRequest(
            name=name,
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            following=following,
        )
        self.user_requests[name] = request
        self._incr_ref_count(name)


        # TODO(Refactor): Poll all of these ready functions inside the
        #  Session instead of hacking in the add request function
        def aici_session__is_request_ready(**kwargs):
            assert following in self.user_requests
            return self.user_requests[following].is_finished()

        def aici_session__request_init_sequence(
            request_id=None,
            seq_id=None,
            processed_inputs=None,
            block_size=None,
            eos_token_id=None,
            lora_request=None,
            **kwargs,
        ):
            # Initialize the sequence
            if not following:
                seq = Sequence(seq_id, processed_inputs, block_size,
                               eos_token_id, lora_request)
                seq.kwargs = dict(aici_session__request_post_finish_hook=
                                  aici_session__request_post_finish_hook)
                return seq

            # Fork from following, then initialize the sequence
            parent = self.user_requests[following].sequence
            assert parent.is_finished()
            child = parent.fork(seq_id)
            self.scheduler.fork_seq(parent, child)
            request.sequence = child
            child.kwargs = dict(aici_session__request_post_finish_hook=
                                aici_session__request_post_finish_hook)
            return child

        def aici_session__register_sequence_group(seq_group, **kwargs) -> None:
            # Register the sequence group into the session
            request.sequence_group = seq_group
            request.sequence_group.kwargs[
                'aici_session__should_pause'] = request.should_pause
            return

        def aici_session__request_post_finish_hook(parent, **kwargs):
            # TODO: _preempt_by_recompute also calls this function.
            #  We should know the context of the call, and decide to fork or recompute?
            # Fork the sequence such that its memory ref is preserved.
            child = parent.fork(
                parent.seq_id +
                10000000)  # TODO: Hack to avoid conflict of hash
            self.scheduler.fork_seq(parent, child)
            request.sequence = child  # Persist the mem reference using the sequence.
            # TODO: backtrack here
            return

        stream = await self.engine.add_request(
            request_id=random_uuid(),
            inputs=prompt,
            params=sampling_params,
            # TODO: (Hack) Custom hook to check the readiness of a request
            aici_session__is_request_ready=aici_session__is_request_ready,
            aici_session__request_init_sequence=
            aici_session__request_init_sequence,
            aici_session__register_sequence_group=
            aici_session__register_sequence_group,
            aici_session__request_post_finish_hook=
            aici_session__request_post_finish_hook,
        )
        if stream_to_user:
            # Delegate the stream to stream handler.
            self.outgoing_streams[name] = stream
        return

    async def pause_sequence(self, name: str):
        request = self.user_requests[name]
        request.should_pause = True
        return

    async def resume_sequence(self, name: str):
        request = self.user_requests[name]
        request.should_pause = False
        return

    async def stop_sequence(self, name: str):
        # abort the request in scheduler,
        # and then actually release the sequence in the engine.
        request = self.user_requests[name]
        # Disable the post finish hook (so it will normally free the sequence at running instead of forking one)
        request.sequence.kwargs.pop('aici_session__request_post_finish_hook',
                                    None)
        self.scheduler.abort_seq_group(request.request_id)
        # Free sequence one more time (will not cause double free problem by how the code is written)
        self.scheduler.free_seq(request.sequence)
        self._decr_ref_count(name)
        return

    async def clean_up(self):
        raise NotImplementedError

    pass


class UserRequest:

    def __init__(
        self,
        name: str,
        request_id=None,
        prompt: str = None,
        prompt_token_ids: List[int] = None,
        following: str = None,
    ):
        self.name = name
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids: List[int] = prompt_token_ids
        self.following: Optional[str] = following
        self.sequence: 'Sequence' = None
        self.sequence_group: 'SequenceGroup' = None

        self._should_pause = False
        pass

    @property
    def should_pause(self):
        return self._should_pause

    @should_pause.setter
    def should_pause(self, value: bool):
        self._should_pause = value
        if self.sequence_group:
            self.sequence_group.kwargs['aici_session__should_pause'] = value
        return

    def is_finished(self) -> bool:
        if self.sequence:
            # TODO: Should this be exclusively "finished" state?
            return self.sequence.is_finished()
        return False
