from typing import Annotated, Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from openhands.app_server.sandbox.session_auth import validate_session_key
from openhands.app_server.user_auth import get_user_id
from openhands.sdk.llm.auth.openai import OPENAI_CODEX_MODELS
from server.auth.org_context import EFFECTIVE_ORG_ID
from server.services.openai_subscription_service import (
    OpenAISubscriptionService,
    PendingLoginNotFound,
)

router = APIRouter(prefix='/api/llm/subscription/openai', tags=['Subscription'])
_service = OpenAISubscriptionService()


class PollRequest(BaseModel):
    device_code: str = Field(min_length=1, max_length=128)


async def _personal_user(
    user_id: str | None = Depends(get_user_id),
    org_id: UUID = EFFECTIVE_ORG_ID,
) -> UUID:
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'Authentication required')
    try:
        user = UUID(user_id)
    except ValueError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, 'Invalid user identity'
        ) from None
    if user != org_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            'Subscription login is available only in a personal workspace',
        )
    return user


UserId = Annotated[UUID, Depends(_personal_user)]


def _no_store(response: Response) -> None:
    response.headers['Cache-Control'] = 'no-store'


@router.get('/models')
async def models(user_id: UserId, response: Response) -> dict[str, Any]:
    _no_store(response)
    return {'vendor': 'openai', 'models': sorted(OPENAI_CODEX_MODELS)}


@router.get('/status')
async def subscription_status(user_id: UserId, response: Response) -> dict[str, Any]:
    _no_store(response)
    try:
        return await _service.status(user_id)
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, 'Subscription provider unavailable'
        ) from None


@router.post('/device/start')
async def start_device_login(user_id: UserId, response: Response) -> dict[str, Any]:
    _no_store(response)
    try:
        return await _service.start(user_id)
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, 'Subscription provider unavailable'
        ) from None


@router.post('/device/poll')
async def poll_device_login(
    body: PollRequest, user_id: UserId, response: Response
) -> dict[str, Any]:
    _no_store(response)
    try:
        return await _service.poll(user_id, body.device_code)
    except PendingLoginNotFound:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, 'Subscription device login not found or expired'
        ) from None
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, 'Subscription provider unavailable'
        ) from None


@router.post('/logout')
async def logout(user_id: UserId, response: Response) -> dict[str, Any]:
    _no_store(response)
    await _service.logout(user_id)
    return {'vendor': 'openai', 'connected': False}


@router.get('/runtime/token', include_in_schema=False)
async def runtime_token(request: Request) -> JSONResponse:
    sandbox = await validate_session_key(request.headers.get('X-Session-API-Key'))
    if not sandbox.created_by_user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'Sandbox has no owner')
    try:
        owner = UUID(sandbox.created_by_user_id)
        workspace = UUID(sandbox.workspace_id or '')
    except ValueError:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, 'Sandbox workspace is not authorized'
        ) from None
    if workspace != owner or not sandbox.subscription_broker_ready:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, 'Sandbox workspace is not authorized'
        )
    try:
        token = await _service.get_runtime_token(owner)
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError, TypeError):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, 'Subscription token unavailable'
        ) from None
    if token is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'Subscription login is required')
    return JSONResponse(
        token, headers={'Cache-Control': 'no-store', 'Pragma': 'no-cache'}
    )
