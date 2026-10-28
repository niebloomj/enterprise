import secrets
import time
from dataclasses import asdict
from typing import Any
from uuid import UUID

from openhands.sdk.llm.auth.credentials import CredentialStore, OAuthCredentials
from openhands.sdk.llm.auth.openai import (
    DEVICE_CODE_TIMEOUT_SECONDS,
    DeviceCode,
    OpenAISubscriptionAuth,
)
from storage.openai_subscription_store import (
    OpenAISubscription,
    OpenAISubscriptionStore,
)


class PendingLoginNotFound(LookupError):
    pass


class _MemoryCredentials(CredentialStore):
    def __init__(self, credentials: OAuthCredentials | None = None) -> None:
        self._credentials = credentials

    def get(self, vendor: str) -> OAuthCredentials | None:
        if self._credentials is None or self._credentials.vendor != vendor:
            return None
        return self._credentials

    def save(self, credentials: OAuthCredentials) -> None:
        self._credentials = credentials


class OpenAISubscriptionService:
    def __init__(self, store: OpenAISubscriptionStore | None = None) -> None:
        self._store = store or OpenAISubscriptionStore()

    async def start(self, user_id: UUID) -> dict[str, Any]:
        async with self._store.locked(user_id) as state:
            auth = OpenAISubscriptionAuth(credential_store=_MemoryCredentials())
            device = await auth.start_device_login()
            token = secrets.token_urlsafe(32)
            expires_at = int(time.time() * 1000) + DEVICE_CODE_TIMEOUT_SECONDS * 1000
            state.pending_login = {
                'token': token,
                'device': asdict(device),
                'expires_at': expires_at,
            }
        return {
            'device_code': token,
            'user_code': device.user_code,
            'verification_uri': device.verification_url,
            'expires_at': expires_at,
            'interval_seconds': device.interval,
        }

    async def poll(self, user_id: UUID, token: str) -> dict[str, Any]:
        async with self._store.locked(user_id) as state:
            pending = state.pending_login
            if (
                pending is None
                or not token.isascii()
                or not secrets.compare_digest(pending['token'], token)
                or pending['expires_at'] <= int(time.time() * 1000)
            ):
                raise PendingLoginNotFound(
                    'Subscription device login not found or expired'
                )
            auth = OpenAISubscriptionAuth(credential_store=_MemoryCredentials())
            credentials = await auth.poll_device_login(
                DeviceCode(**pending['device']), persist=False
            )
            if credentials is None:
                return {'vendor': 'openai', 'connected': False}
            state.credentials = credentials.model_dump()
            state.pending_login = None
            return {
                'vendor': 'openai',
                'connected': True,
                'expires_at': credentials.expires_at,
            }

    async def _current_credentials(
        self, state: OpenAISubscription
    ) -> OAuthCredentials | None:
        if state.credentials is None:
            return None
        stored = OAuthCredentials.model_validate(state.credentials)
        auth = OpenAISubscriptionAuth(credential_store=_MemoryCredentials(stored))
        credentials = await auth.refresh_if_needed()
        if credentials is not None and credentials != stored:
            state.credentials = credentials.model_dump()
        return credentials

    async def status(self, user_id: UUID) -> dict[str, Any]:
        async with self._store.locked(user_id) as state:
            credentials = await self._current_credentials(state)
            return {
                'vendor': 'openai',
                'connected': credentials is not None,
                'expires_at': credentials.expires_at if credentials else None,
            }

    async def get_runtime_token(self, user_id: UUID) -> dict[str, Any] | None:
        async with self._store.locked(user_id) as state:
            credentials = await self._current_credentials(state)
            if credentials is None:
                return None
            if credentials.is_expired():
                raise RuntimeError('Subscription provider returned an expired token')
            return {
                'access_token': credentials.access_token,
                'expires_at': credentials.expires_at,
            }

    async def get_access_token(self, user_id: UUID) -> str | None:
        token = await self.get_runtime_token(user_id)
        return token['access_token'] if token else None

    async def logout(self, user_id: UUID) -> None:
        async with self._store.locked(user_id) as state:
            state.credentials = None
            state.pending_login = None
