"""
gRPC Policy server leveraging batch inputs.
"""

import argparse
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import dataclasses as dc
import time
import uuid

import grpc
import numpy as np

from grpc_workspace.lbm_policy_conversions import (
    grpc_msg_to_policy_observation,
    grpc_msg_to_uuid,
    policy_metadata_to_grpc_msg,
    poses_and_grippers_to_grpc_msg,
)
from grpc_workspace.proto import (
    GetPolicyMetadata_pb2,
    GetPolicyMetadata_pb2_grpc,
    PolicyReset_pb2,
    PolicyReset_pb2_grpc,
    PolicyStep_pb2,
    PolicyStep_pb2_grpc,
    health_pb2,
    health_pb2_grpc,
)
from robot_gym.multiarm_spaces import (
    MultiarmObservation,
    PosesAndGrippers,
)
from robot_gym.policy import Policy

# TODO(imcmahon): Consider consolidating each of these
# services into a single proto service like the grpc examples
# https://github.com/grpc/grpc/blob/master/examples/protos/route_guide.proto
# https://github.com/grpc/grpc/blob/master/examples/python/route_guide/asyncio_route_guide_server.py


class HealthService(health_pb2_grpc.HealthServicer):
    def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.ServicerContext,
    ) -> health_pb2.HealthCheckResponse:
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING
        )


# At this stage LbmPolicyServer is not defined, so we must use a string
# literal version in its place. See the Type Hints PEP for more details:
# https://peps.python.org/pep-0484/#forward-references
class PolicyStepService(PolicyStep_pb2_grpc.PolicyStepServiceServicer):
    def __init__(self, lbm_policy_server: "LbmPolicyServer"):
        self._lbm_policy_server = lbm_policy_server

    async def PolicyStep(
        self,
        request: PolicyStep_pb2.PolicyStepRequest,
        context: grpc.ServicerContext,
    ) -> PolicyStep_pb2.PolicyStepResponse:
        action, success = await self._lbm_policy_server._step(request)
        return PolicyStep_pb2.PolicyStepResponse(
            action=action,
            success=success,
        )


class PolicyResetService(PolicyReset_pb2_grpc.PolicyResetServiceServicer):
    def __init__(self, lbm_policy_server: "LbmPolicyServer"):
        self._lbm_policy_server = lbm_policy_server

    async def PolicyReset(
        self,
        request: PolicyReset_pb2.PolicyResetRequest,
        context: grpc.ServicerContext,
    ) -> PolicyReset_pb2.PolicyResetResponse:
        success = await self._lbm_policy_server._reset(request)
        return PolicyReset_pb2.PolicyResetResponse(success=success)


class GetPolicyMetadataService(
    GetPolicyMetadata_pb2_grpc.GetPolicyMetadataServiceServicer
):
    def __init__(self, lbm_policy_server: "LbmPolicyServer"):
        self._lbm_policy_server = lbm_policy_server

    async def GetPolicyMetadata(
        self,
        request: GetPolicyMetadata_pb2.GetPolicyMetadataRequest,
        context: grpc.ServicerContext,
    ) -> GetPolicyMetadata_pb2.GetPolicyMetadataResponse:
        (
            policy_metadata,
            success,
        ) = await self._lbm_policy_server._get_policy_metadata(request)
        return GetPolicyMetadata_pb2.GetPolicyMetadataResponse(
            policy_metadata=policy_metadata,
            success=success,
        )


# Calculation for thirty megabytes.
THIRTY_MB = 30 * 1024 * 1024


@dc.dataclass(kw_only=True)
class LbmPolicyServerConfig:
    """
    Args:
        batch_timeout_s (float): A positive, non-zero float representing the
            maximum number of seconds to wait between successive calls to the
            Policy.
        batch_max_size (int): A positive, non-zero integer representing the
            maximum number of observations to batch together when calling the
            Policy.
        server_uri (str): A URI of the format 'address:port' to indicate
            the address of the gRPC server.
        grpc_max_send_message_length (int): Maximum message length the gRPC
            channels can send.
        grpc_max_receive_message_length (int): Maximum message length the gRPC
            channels can receive.
    """

    batch_timeout_s: float = 0.5
    batch_max_size: int = 1
    # TODO(imcmahon): Create a common library for both server and client to
    # use as the defaults for the server configurations.
    server_uri: str = "localhost:50051"
    # TODO(charlie): Determine appropriate limits.
    # Set 30 MB limit for send and receive for now.
    grpc_max_send_message_length: int = THIRTY_MB
    grpc_max_receive_message_length: int = THIRTY_MB

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.batch_max_size < 1:
            raise ValueError(
                "batch_max_size must be an integer 1 or greater: "
                f"{self.batch_max_size}"
            )

    def create(self, policy: Policy):
        """
        Args:
            policy (Policy): The supplied Policy to be called inside a
                LbmPolicyServer.
        """
        self.validate()
        return LbmPolicyServer(policy=policy, **vars(self))

    @staticmethod
    def from_argparse_args(
        args: argparse.Namespace | None = None,
    ):
        defaults = LbmPolicyServerConfig()
        policy_server_kwargs = dict()
        for field in dc.fields(LbmPolicyServerConfig):
            default_value = getattr(defaults, field.name)
            # Re-applying a default value here is probably redundant given that
            # users will have used `add_argparse_args()` to begin with, but we
            # are slightly more robust this way.
            args_value = getattr(args, field.name, default_value)
            policy_server_kwargs[field.name] = args_value
        return LbmPolicyServerConfig(**policy_server_kwargs)

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser):

        def add_argument(arg_str: str, arg_type: type, help_str: str):
            parser.add_argument(
                "--{}".format(arg_str.replace("_", "-")),
                dest=arg_str,
                default=getattr(LbmPolicyServerConfig, arg_str),
                required=False,
                type=arg_type,
                help=help_str,
            )

        # NOTE(dale.mcconachie): We are using `_` in the argument names
        # here but then immediately overriding them to `-` in
        # `add_argument` to be consistent with lbm's style. We do it this
        # way so that we can lookup the argument names, types, and docstrings.
        add_argument(
            arg_str="batch_timeout_s",
            arg_type=float,
            help_str=(
                "A positive, non-zero float representing the maximum number "
                "of seconds to wait between successive calls to the Policy."
            ),
        )
        add_argument(
            arg_str="batch_max_size",
            arg_type=int,
            help_str=(
                "A positive, non-zero integer representing the maximum number "
                "of observations to batch together when calling a Policy."
            ),
        )
        add_argument(
            arg_str="server_uri",
            arg_type=str,
            help_str=(
                "A URI of the format 'address:port' to indicate the address "
                "of the gRPC server."
            ),
        )
        add_argument(
            arg_str="grpc_max_send_message_length",
            arg_type=int,
            help_str="Maximum message length the gRPC channels can send.",
        )
        add_argument(
            arg_str="grpc_max_receive_message_length",
            arg_type=int,
            help_str="Maximum message length the gRPC channels can receive.",
        )


class LbmPolicyServer:
    def __init__(
        self,
        policy: Policy,
        batch_timeout_s: float,
        batch_max_size: int,
        server_uri: str,
        grpc_max_send_message_length: int,
        grpc_max_receive_message_length: int,
    ):
        self._policy = policy
        self._batch_timeout_s = batch_timeout_s
        self._batch_max_size = batch_max_size
        self._server_uri = server_uri
        self._server_options = [
            ("grpc.max_send_message_length", grpc_max_send_message_length),
            (
                "grpc.max_receive_message_length",
                grpc_max_receive_message_length,
            ),
        ]

        # Mutex lock for accessing every batch variable that starts with
        # "_reset" including "_reset_mailbox_inputs",
        # "_reset_mailbox_responses".
        self._reset_mailbox_lock = asyncio.Lock()
        # The seeds are represented by uint32 values over RPC, defined by
        # grpc_workspace/proto/PolicyReset.proto. See the following links
        # for details:
        # https://drake.mit.edu/pydrake/pydrake.common.html?highlight=random#pydrake.common.RandomGenerator
        # https://en.cppreference.com/w/cpp/numeric/random/mersenne_twister_engine
        # TODO(imcmahon): Decide on the appropriate int type. See lbm#1359.
        self._reset_mailbox_inputs: dict[uuid.UUID, np.uint32] = dict()
        self._reset_mailbox_responses: list[uuid.UUID] = list()

        # Mutex lock for accessing every batch variable that starts with
        # "_step" including "_step_last_time", "_step_mailbox_inputs",
        # and "_step_mailbox_responses".
        self._step_mailbox_lock = asyncio.Lock()
        self._step_last_time = time.time()
        self._step_mailbox_inputs: OrderedDict[
            uuid.UUID,
            MultiarmObservation,
        ] = OrderedDict()
        self._step_mailbox_responses: dict[
            uuid.UUID,
            PosesAndGrippers,
        ] = dict()

        # Guarantees that the _loop_once cannot be reentrant.
        self._loop_once_lock = asyncio.Lock()

        # Add gRPC Services.
        # TODO(imcmahon): Store client identifiers and use them to flow control
        # clients ensuring that "reset" is always called before "step" the
        # first time a client identifier is encountered.
        self._policy_step_servicer = PolicyStepService(lbm_policy_server=self)
        self._policy_reset_servicer = PolicyResetService(
            lbm_policy_server=self,
        )
        self._get_policy_metadata_servicer = GetPolicyMetadataService(
            lbm_policy_server=self,
        )
        self._health_servicer = HealthService()

        # This executor is designated for calling the Policy's functions.
        # One worker should be sufficient, as the policy itself is not
        # expected to be thread-safe. However, the policy's
        # get_policy_metadata() should be thread-safe.
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def _is_reset_client_pending(self, client_id: uuid.UUID) -> bool:
        async with self._reset_mailbox_lock:
            is_pending = client_id in self._reset_mailbox_inputs
        return is_pending

    async def _reset(
        self, request: PolicyReset_pb2.PolicyResetRequest
    ) -> bool:
        print(
            f"gRPC client 'reset' request received with seed {request.seed}."
        )
        success = False
        client_id = grpc_msg_to_uuid(request.client_identifier)
        async with self._reset_mailbox_lock:
            # Guard against a client making multiple reset requests
            # simultaneously.
            if client_id in self._reset_mailbox_responses:
                return False
            if client_id in self._reset_mailbox_inputs:
                return False
            self._reset_mailbox_inputs[client_id] = np.uint32(request.seed)

        # Check the input batch status until the request is processed.
        while await self._is_reset_client_pending(client_id):
            await asyncio.sleep(0.01)

        # Check pending requests for client id.
        async with self._reset_mailbox_lock:
            try:
                # If this excepts, it likely means the client has done
                # something unexpected like simultaneous calls to step and
                # reset or two clients are using the same UUID.
                # TODO(imcmahon): Make the `success` value an optional string
                # to better report the kind of error encountered.
                self._reset_mailbox_responses.remove(client_id)
                success = True
            except ValueError:
                success = False
        return success

    async def _is_batch_client_pending(self, client_id: uuid.UUID) -> bool:
        async with self._step_mailbox_lock:
            is_pending = client_id in self._step_mailbox_inputs
        return is_pending

    async def _step(
        self,
        request: PolicyStep_pb2.PolicyStepRequest,
    ) -> tuple[PolicyStep_pb2.PosesAndGrippers, bool]:
        response = None
        success = False
        # success == False shouldn't happen, but if it somehow does, report an
        # empty PosesAndGrippers protobuf object alongside success = False to
        # the client.
        action = PolicyStep_pb2.PosesAndGrippers()
        client_id = grpc_msg_to_uuid(request.client_identifier)
        # Convert gRPC to MultiarmObservation and store it in batch_inputs.
        policy_observation = grpc_msg_to_policy_observation(
            request.observation,
        )
        async with self._step_mailbox_lock:
            # Guard against a client making multiple step requests
            # simultaneously.
            if client_id in self._step_mailbox_responses:
                return action, False
            if client_id in self._step_mailbox_inputs:
                return action, False
            self._step_mailbox_inputs[client_id] = policy_observation

        # Check the input batch status until the request is processed.
        while await self._is_batch_client_pending(client_id):
            await asyncio.sleep(0.01)

        # Check pending requests and convert the action to gRPC message.
        async with self._step_mailbox_lock:
            # If this excepts, it likely means the client has done something
            # unexpected like simultaneous calls to step and reset or
            # two clients are using the same UUID.
            # TODO(imcmahon): Make the `success` value an optional string
            # to better report the kind of error encountered.
            try:
                response = self._step_mailbox_responses.pop(client_id)
                success = True
            except KeyError:
                success = False
        if success:
            action = poses_and_grippers_to_grpc_msg(response)
        return action, success

    async def _get_policy_metadata(
        self, request: GetPolicyMetadata_pb2.GetPolicyMetadataRequest
    ) -> tuple[GetPolicyMetadata_pb2.PolicyMetadata, bool]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._policy.get_policy_metadata,
        )
        # Convert the PolicyMetadata to gRPC message.
        policy_metadata_msg = policy_metadata_to_grpc_msg(result)
        return policy_metadata_msg, True

    def _raise_on_list_intersection(self, first_list, second_list):
        overlap = set(first_list).intersection(second_list)
        if len(overlap):
            raise RuntimeError(f"Overlap in lists detected: {overlap}")

    def _get_and_update_step_inputs(self):
        # Assumes caller is holding the self._step_mailbox_lock.
        # TODO(imcmahon): Enforce this assumption in code.
        if len(self._step_mailbox_inputs) > self._batch_max_size:
            step_inputs_dict = dict()
            for _ in range(self._batch_max_size):
                # First-in-first-out pop ordering for selective removal.
                (key, value) = self._step_mailbox_inputs.popitem(last=False)
                step_inputs_dict[key] = value
        else:
            step_inputs_dict = dict(self._step_mailbox_inputs)
            self._step_mailbox_inputs.clear()
        return step_inputs_dict

    async def _process_reset(self):
        async with self._reset_mailbox_lock:
            if len(self._reset_mailbox_inputs.values()):
                self._raise_on_list_intersection(
                    self._reset_mailbox_inputs,
                    self._reset_mailbox_responses,
                )
                # TODO(imcmahon): Investigate running the policy's
                # reset_batch thead without holding the _reset_mailbox_lock
                # to allow gRPC reset() callbacks to be serviced while
                # reset is happening.
                # Dispatching reset_batch to a thread is not necessarily
                # required for our current policies as reset_batch should have
                # minimal work being done, but that may not be true of future
                # polices, and using a threaded approach gains us some
                # robustness.
                await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._policy.reset_batch,
                    self._reset_mailbox_inputs,
                )
                self._reset_mailbox_responses.extend(
                    self._reset_mailbox_inputs.keys(),
                )
                self._reset_mailbox_inputs.clear()

    async def _process_step(self):
        async with self._step_mailbox_lock:
            current_time = time.time()
            elapsed_time = current_time - self._step_last_time
            # Check if batch processing requirements are met and store
            # the relevant client identifier-observations in a dictionary
            # to be handed to the policy.
            batch_size = len(self._step_mailbox_inputs)
            should_process_max_batch = batch_size >= self._batch_max_size
            should_process_batch_timeout = (
                batch_size > 0 and elapsed_time >= self._batch_timeout_s
            )
            if should_process_max_batch or should_process_batch_timeout:
                self._raise_on_list_intersection(
                    list(self._step_mailbox_inputs.keys()),
                    list(self._step_mailbox_responses.keys()),
                )
                obs_dict = self._get_and_update_step_inputs()

                # TODO(imcmahon): Investigate running the policy's
                # step_batch thead without holding the _step_mailbox_lock
                # to allow gRPC step() callbacks to be serviced while
                # inference is happening.
                # TODO(dale.mcconachie) If this call fails, we simply get a
                # "RuntimeError: The server rejected this client uuid" on the
                # client side with no other information or logging. We should
                # improve this.
                step_batch_result = (
                    await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self._policy.step_batch,
                        obs_dict,
                    )
                )
                self._step_mailbox_responses.update(step_batch_result)
                self._step_last_time = current_time

    async def _loop_once(self):
        # Lock around a critical section to prevent reentry. This is done as
        # _loop_once would break if it were ever reentrant, potentially by a
        # future developer calling `_loop_once` from two different spots.
        # This function enforces that `_process_reset` and `_process_step` are
        # called sequentially in the desired ordering, i.e. reset then step.
        async with self._loop_once_lock:
            await self._process_reset()
            await self._process_step()

    async def start(self):
        # The `start` function utilizes asyncio to run the gRPC server and all
        # of its callbacks in the same main thread. This is done by yielding
        # with `await` whenever a blocking call is made, guarding data and
        # critical sections with non-thread-safe asyncio.Lock()'s.
        # The Policy code is not guaranteed to conform to this asyncio
        # paradigm, and calls to it are generally run in a ThreadPoolExecutor,
        # and awaited. This allows Policy code ignore the fact that the rest of
        # this server uses an asyncio event loop.
        server = grpc.aio.server(options=self._server_options)

        PolicyStep_pb2_grpc.add_PolicyStepServiceServicer_to_server(
            self._policy_step_servicer, server
        )
        PolicyReset_pb2_grpc.add_PolicyResetServiceServicer_to_server(
            self._policy_reset_servicer, server
        )
        GetPolicyMetadata_pb2_grpc.add_GetPolicyMetadataServiceServicer_to_server(  # noqa
            self._get_policy_metadata_servicer, server
        )
        health_pb2_grpc.add_HealthServicer_to_server(
            self._health_servicer, server
        )
        server.add_insecure_port(self._server_uri)
        await server.start()

        print(f"Started Server loop on {self._server_uri}...", flush=True)
        try:
            # TODO(imcmahon) Look into using asyncio.Events instead of the
            # while `condition` async loops in this server.
            while True:
                await self._loop_once()
                await asyncio.sleep(0.01)
        finally:
            await server.stop(1.0)


def run_policy_server(policy: Policy, args: argparse.Namespace | None = None):
    """
    Args:
        policy (Policy): The supplied Policy to be called inside a
            LbmPolicyServer.
        args (argparse.Namespace): An optional namespace constructed by
            argparse. If provided, this will be used in conjunction with the
            policy argument to construct a LbmPolicyServer object from
            LbmPolicyServerConfig.add_argparse_arguments().
            If None is provided, the default values in LbmPolicyServerConfig
            will be used in conjunction with the policy argument in
            constructing the LbmPolicyServer object.
    """
    # TODO(dale.mcconachie): We should be able to configure the policy itself
    # via YAML.
    lbm_policy_server_config = LbmPolicyServerConfig.from_argparse_args(args)
    lbm_policy_server = lbm_policy_server_config.create(policy)
    asyncio.run(lbm_policy_server.start())
