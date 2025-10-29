import dataclasses as dc
import logging
from typing import Any
import uuid

import grpc
import numpy as np

from grpc_workspace.lbm_policy_conversions import (
    grpc_msg_to_policy_metadata,
    grpc_msg_to_poses_and_grippers,
    policy_observation_to_grpc_msg,
    uuid_to_grpc_msg,
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
from robot_gym.multiarm_spaces import MultiarmObservation
from robot_gym.policy import Policy, PolicyConfig


@dc.dataclass
class LbmPolicyClientConfig(PolicyConfig):
    """
    Args:
        wait_for_server (bool): True indicates that the initialization
            of this client should block until a server is detected.
        server_uri (str): A URI of the format 'address:port' to indicate
            the location of the gRPC server.
        grpc_max_send_message_length (int): Maximum message length the gRPC
            channels can send.
        grpc_max_receive_message_length (int): Maximum message length the gRPC
            channels can receive.
    """

    wait_for_server: bool = True
    # TODO(imcmahon): Create a common library for both server and client to
    # use as the defaults for the server configurations.
    server_uri: str = "localhost:50051"
    # TODO(charlie): Determine appropriate limits.
    # Set 30 MB limit for send and receive for now.
    grpc_max_send_message_length: int = 30 * 1024 * 1024
    grpc_max_receive_message_length: int = 30 * 1024 * 1024

    def create(self):
        # TODO(imcmahon): Consider using dict unpacking rather than having the
        # LbmPolicyClient object take one large config which it then has to
        # index into every time it uses a value from the config.
        return LbmPolicyClient(config=self)


class _UuidLogger(logging.LoggerAdapter):
    """Adds a UUID prefix to all log messages."""

    def __init__(self, *, logger, uuid):
        super().__init__(logger, None)
        self._uuid = uuid

    def process(self, old_message, kwargs):
        new_message = f"[{self._uuid}] {old_message}"
        return (new_message, kwargs)


class LbmPolicyClient(Policy):
    """A policy that communicates with a gRPC server to compute actions.

    This policy converts requests into gRPC messages adhering to
    specific service interfaces (PolicyStep, PolicyReset, and
    GetPolicyMetadata), and sends that message over RPC to a gRPC server
    running in a separate process. That separate process replies over RPC which
    this policy formats and returns.

    This class sends every observation to the server rather than handling the
    stacking of observations and unrolling of actions here. The server and its
    associated policy should handle any stacking of observations and unrolling
    of actions.
    """

    def __init__(self, config: LbmPolicyClientConfig):

        assert isinstance(config, LbmPolicyClientConfig)

        self._server_uri = config.server_uri
        self._grpc_options = [
            (
                "grpc.max_send_message_length",
                config.grpc_max_send_message_length,
            ),
            (
                "grpc.max_receive_message_length",
                config.grpc_max_receive_message_length,
            ),
        ]
        self._uuid = uuid.uuid4()
        self._client_identifier_msg = uuid_to_grpc_msg(self._uuid)

        self._logger = _UuidLogger(
            logger=logging.getLogger("lbm_policy_client"),
            uuid=self._uuid,
        )

        # TODO(imcmahon): Add semantics for a "verbose" wait_for_ready
        # on every Request. This should include a loop of calls to
        # the Request with wait_for_ready=True and a relatively short
        # (1-10 second) timeout set. After the timeout expires, print
        # logging information about the connection, then loop back to
        # another Request with wait_for_ready=True.
        if config.wait_for_server:
            # TODO(imcmahon): Find a way to improve the readability for all of
            # the gRPC insecure channel connections.
            with grpc.insecure_channel(
                self._server_uri,
                options=self._grpc_options,
            ) as channel:
                self._wait_for_server(channel)

    def _wait_for_server(self, channel: grpc.ServicerContext):
        self._logger.info(f"Waiting for gRPC server on {self._server_uri} ...")
        health_stub = health_pb2_grpc.HealthStub(channel)
        response = health_stub.Check(
            health_pb2.HealthCheckRequest(service=""),
            wait_for_ready=True,
        )
        if response.status == health_pb2.HealthCheckResponse.SERVING:
            self._logger.info("... gRPC server is ready!")
        else:
            raise RuntimeError(f"Unable to connect to server: {response}")

    def _success_or_throw(self, response):
        if not response.success:
            raise RuntimeError(
                f"The server rejected this client uuid: {self._uuid}."
            )

    def get_policy_metadata(self):
        req = GetPolicyMetadata_pb2.GetPolicyMetadataRequest(
            client_identifier=self._client_identifier_msg,
        )
        with grpc.insecure_channel(
            self._server_uri,
            options=self._grpc_options,
        ) as channel:
            stub = GetPolicyMetadata_pb2_grpc.GetPolicyMetadataServiceStub(
                channel,
            )
            response = stub.GetPolicyMetadata(req, wait_for_ready=True)
        self._success_or_throw(response)
        return grpc_msg_to_policy_metadata(response.policy_metadata)

    def reset(
        self,
        *,
        seed: np.uint32 | None = None,
        options: dict[str, Any] = None,
    ):
        # Note: This function signature is meant to (mostly) conform to
        # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset.
        # The `options` argument is not currently used, and not associated
        # with the gRPC insecure_channel `options` argument.
        req = PolicyReset_pb2.PolicyResetRequest(
            client_identifier=self._client_identifier_msg,
            seed=seed,
        )
        with grpc.insecure_channel(
            self._server_uri,
            options=self._grpc_options,
        ) as channel:
            # TODO(imcmahon) Profile/test what happens if wait_for_ready=True
            # is omitted on calls beyond the first. See the following reference
            # for details: https://grpc.io/docs/guides/wait-for-ready/
            stub = PolicyReset_pb2_grpc.PolicyResetServiceStub(channel)
            response = stub.PolicyReset(req, wait_for_ready=True)
        self._success_or_throw(response)

    def step(self, observation: MultiarmObservation):
        req = PolicyStep_pb2.PolicyStepRequest(
            client_identifier=self._client_identifier_msg,
            observation=policy_observation_to_grpc_msg(observation),
        )

        with grpc.insecure_channel(
            self._server_uri,
            options=self._grpc_options,
        ) as channel:
            stub = PolicyStep_pb2_grpc.PolicyStepServiceStub(channel)
            response = stub.PolicyStep(req, wait_for_ready=True)

        self._success_or_throw(response)
        return grpc_msg_to_poses_and_grippers(response.action)


# Alias the namechange from lbm#1450 for lbm_eval_0_5 backward compatibility.
# TODO(imcmahon): Remove these aliases once Anzu's lbm_eval_0_5 branch is no
# longer testing compatibility against LBM main in Jenkins CI's
# Anzu-branches-LBM-main build.
LbmClientPolicyConfig = LbmPolicyClientConfig
LbmClientPolicy = LbmPolicyClient
