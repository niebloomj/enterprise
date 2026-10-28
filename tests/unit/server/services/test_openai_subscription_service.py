import asyncio
import socket
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
import uvicorn
from aiohttp import web
from fastapi import FastAPI
from pydantic import SecretStr

from openhands.app_server.errors import AuthError
from openhands.app_server.sandbox import session_auth
from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)
from openhands.app_server.sandbox.remote_sandbox_service import RemoteSandboxService
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.sandbox.sandbox_store import (
    StoredSandbox,
    hash_session_api_key,
)
from openhands.app_server.shared import server_config
from openhands.app_server.user.auth_user_context import AuthUserContext
from openhands.app_server.user.specifiy_user_context import ADMIN
from openhands.app_server.user.user_models import UserInfo
from openhands.sdk.llm.auth import credentials as sdk_credentials
from openhands.sdk.llm.auth import openai
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.openai_subscription import router
from server.services.openai_subscription_service import OpenAISubscriptionService
from storage.api_key_store import ApiKeyStore
from storage.openai_subscription_store import OpenAISubscriptionStore


@pytest.fixture
async def provider(monkeypatch):
    calls = {
        'ready': False,
        'refreshes': 0,
        'exchanges': 0,
        'start_gate': None,
        'malformed_token': False,
        'start_entered': asyncio.Event(),
    }

    async def start(request):
        calls['start_entered'].set()
        if calls['start_gate'] is not None:
            await calls['start_gate'].wait()
        return web.json_response(
            {
                'device_auth_id': 'provider-device',
                'user_code': 'ABCD-1234',
                'interval': 1,
            }
        )

    async def poll(request):
        if not calls['ready']:
            return web.Response(status=403)
        return web.json_response(
            {'authorization_code': 'code', 'code_verifier': 'verifier'}
        )

    async def token(request):
        data = await request.post()
        if data['grant_type'] == 'refresh_token':
            calls['refreshes'] += 1
            assert data['refresh_token'] == 'refresh-1'
            await asyncio.sleep(0.1)
            return web.json_response(
                {
                    'access_token': 'access-2',
                    'refresh_token': 'refresh-2',
                    'expires_in': 3600,
                }
            )
        calls['exchanges'] += 1
        if calls['malformed_token']:
            return web.json_response(
                {
                    'access_token': {'sensitive': 'private-provider-value'},
                    'expires_in': 1,
                }
            )
        return web.json_response(
            {'access_token': 'access-1', 'refresh_token': 'refresh-1', 'expires_in': 1}
        )

    app = web.Application()
    app.router.add_post('/api/accounts/deviceauth/usercode', start)
    app.router.add_post('/api/accounts/deviceauth/token', poll)
    app.router.add_post('/oauth/token', token)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setattr(openai, 'ISSUER', f'http://127.0.0.1:{port}')
    try:
        yield calls
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_login_is_user_isolated_and_logout_revokes_state(
    provider, create_user, async_session_maker
):
    first, second = create_user(), create_user()
    service = OpenAISubscriptionService(OpenAISubscriptionStore(async_session_maker))
    challenge = await service.start(first.id)
    assert challenge['user_code'] == 'ABCD-1234'
    assert 'provider-device' not in repr(challenge)
    assert not (await service.poll(first.id, challenge['device_code']))['connected']
    with pytest.raises(LookupError):
        await service.poll(second.id, challenge['device_code'])
    with pytest.raises(LookupError):
        await service.poll(first.id, '☃')
    provider['ready'] = True
    assert (await service.poll(first.id, challenge['device_code']))['connected']
    assert not (await service.status(second.id))['connected']
    assert (await service.get_access_token(first.id)) == 'access-2'
    assert (await service.status(first.id))['connected']
    await service.logout(first.id)
    assert await service.get_access_token(first.id) is None
    assert not (await service.status(first.id))['connected']
    with pytest.raises(LookupError):
        await service.poll(first.id, challenge['device_code'])


@pytest.mark.asyncio
async def test_refresh_is_serialized_and_rotated_token_is_persisted(
    provider, create_user, async_session_maker
):
    user = create_user()
    store = OpenAISubscriptionStore(async_session_maker)
    service = OpenAISubscriptionService(store)
    challenge = await service.start(user.id)
    provider['ready'] = True
    await service.poll(user.id, challenge['device_code'])
    tokens = await asyncio.gather(
        service.get_access_token(user.id), service.get_access_token(user.id)
    )
    assert tokens == ['access-2', 'access-2']
    assert provider['refreshes'] == 1
    async with store.locked(user.id) as state:
        assert state.credentials['refresh_token'] == 'refresh-2'


@pytest.mark.asyncio
async def test_logout_waits_for_in_flight_start_and_invalidates_it(
    provider, create_user, async_session_maker
):
    user = create_user()
    service = OpenAISubscriptionService(OpenAISubscriptionStore(async_session_maker))
    provider['start_gate'] = asyncio.Event()
    start = asyncio.create_task(service.start(user.id))
    await asyncio.wait_for(provider['start_entered'].wait(), timeout=5)
    logout = asyncio.create_task(service.logout(user.id))
    await asyncio.sleep(0.1)
    assert not logout.done()
    provider['start_gate'].set()
    challenge = await asyncio.wait_for(start, timeout=5)
    await asyncio.wait_for(logout, timeout=5)
    provider['ready'] = True
    with pytest.raises(LookupError):
        await service.poll(user.id, challenge['device_code'])
    assert provider['exchanges'] == 0


@pytest.mark.asyncio
async def test_malformed_provider_response_is_classified_without_sensitive_details(
    provider, create_user, create_org, app_db_session, monkeypatch, caplog
):
    monkeypatch.setattr(
        server_config, 'user_auth_class', 'server.auth.saas_user_auth.SaasUserAuth'
    )
    user_id = uuid4()
    create_org(id=user_id)
    create_user(id=user_id, current_org_id=user_id)
    key = await ApiKeyStore().create_api_key(str(user_id), org_id=user_id)
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url='http://test',
    ) as client:
        headers = {'X-Access-Token': key}
        start = await client.post(
            '/api/llm/subscription/openai/device/start', headers=headers
        )
        provider['ready'] = True
        provider['malformed_token'] = True
        response = await client.post(
            '/api/llm/subscription/openai/device/poll',
            headers=headers,
            json={'device_code': start.json()['device_code']},
        )
    assert response.status_code == 503
    assert 'private-provider-value' not in response.text + caplog.text


@pytest.mark.asyncio
async def test_personal_api_uses_authenticated_identity_and_hides_tokens(
    provider, create_user, create_org, app_db_session, monkeypatch
):
    monkeypatch.setattr(
        server_config, 'user_auth_class', 'server.auth.saas_user_auth.SaasUserAuth'
    )
    first_id, second_id = uuid4(), uuid4()
    create_org(id=first_id)
    create_org(id=second_id)
    team = create_org()
    create_user(id=first_id, current_org_id=first_id)
    create_user(id=second_id, current_org_id=second_id)
    keys = ApiKeyStore()
    first_key = await keys.create_api_key(str(first_id), org_id=first_id)
    second_key = await keys.create_api_key(str(second_id), org_id=second_id)
    team_key = await keys.create_api_key(str(first_id), org_id=team.id)
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:

        def headers(key):
            return {'X-Access-Token': key}

        start = await client.post(
            '/api/llm/subscription/openai/device/start', headers=headers(first_key)
        )
        assert start.status_code == 200
        challenge = start.json()['device_code']
        assert start.headers['cache-control'] == 'no-store'
        other_poll = await client.post(
            '/api/llm/subscription/openai/device/poll',
            headers=headers(second_key),
            json={'device_code': challenge},
        )
        assert other_poll.status_code == 404
        assert (
            await client.get(
                '/api/llm/subscription/openai/status', headers=headers(team_key)
            )
        ).status_code == 403
        provider['ready'] = True
        poll = await client.post(
            '/api/llm/subscription/openai/device/poll',
            headers=headers(first_key),
            json={'device_code': challenge},
        )
        assert poll.status_code == 200 and poll.json()['connected']
        status = await client.get(
            '/api/llm/subscription/openai/status', headers=headers(first_key)
        )
        assert status.status_code == 200 and status.json()['connected']
        assert not (
            await client.get(
                '/api/llm/subscription/openai/status', headers=headers(second_key)
            )
        ).json()['connected']
        assert all(
            secret not in start.text + poll.text + status.text
            for secret in (
                'provider-device',
                'access-1',
                'access-2',
                'refresh-1',
                'refresh-2',
            )
        )
        logout = await client.post(
            '/api/llm/subscription/openai/logout', headers=headers(first_key)
        )
        assert logout.status_code == 200 and not logout.json()['connected']


@pytest.mark.asyncio
async def test_runtime_broker_uses_real_sandbox_identity_and_running_status(
    create_user, async_session_maker, app_db_session, monkeypatch
):
    first, second = create_user(), create_user()
    states = {
        name: 'running'
        for name in ('sb-first', 'sb-second', 'sb-team', 'sb-legacy', 'sb-old')
    }
    keys = {name: f'{name}-session-key' for name in states}
    store = OpenAISubscriptionStore(async_session_maker)
    for user, access in (
        (first, 'first-private-token'),
        (second, 'second-private-token'),
    ):
        async with store.locked(user.id) as subscription:
            subscription.credentials = {
                'vendor': 'openai',
                'access_token': access,
                'refresh_token': 'refresh-private',
                'expires_at': int(time.time() * 1000) + 3600000,
            }
    async with async_session_maker() as session, session.begin():
        for sandbox_id, user, workspace_id in (
            ('sb-first', first, str(first.id)),
            ('sb-second', second, str(second.id)),
            ('sb-team', first, str(uuid4())),
            ('sb-legacy', first, None),
            ('sb-old', first, str(first.id)),
        ):
            session.add(
                StoredSandbox(
                    id=sandbox_id,
                    backend='remote',
                    created_by_user_id=str(user.id),
                    workspace_id=workspace_id,
                    subscription_broker_ready=sandbox_id != 'sb-old',
                    sandbox_spec_id='default',
                    session_api_key_hash=hash_session_api_key(keys[sandbox_id]),
                )
            )

    async def runtime(request):
        sandbox_id = request.match_info['sandbox_id']
        return web.json_response(
            {'status': states[sandbox_id], 'session_api_key': keys[sandbox_id]}
        )

    runtime_app = web.Application()
    runtime_app.router.add_get('/sessions/{sandbox_id}', runtime)
    runner = web.AppRunner(runtime_app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    app = FastAPI()
    app.include_router(router)
    try:
        async with httpx.AsyncClient() as runtime_client:

            @asynccontextmanager
            async def sandbox_service(_state):
                async with async_session_maker() as session:
                    yield RemoteSandboxService(
                        sandbox_spec_service=None,
                        api_url=f'http://127.0.0.1:{port}',
                        api_key='runtime-key',
                        web_url=None,
                        resource_factor=1,
                        runtime_class=None,
                        start_sandbox_timeout=10,
                        max_num_sandboxes=1,
                        user_context=ADMIN,
                        httpx_client=runtime_client,
                        db_session=session,
                    )

            monkeypatch.setattr(session_auth, 'get_sandbox_service', sandbox_service)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url='http://test'
            ) as client:
                path = '/api/llm/subscription/openai/runtime/token'
                assert (await client.get(path)).status_code == 401
                assert (
                    await client.get(path, headers={'X-Session-API-Key': 'wrong'})
                ).status_code == 401
                first_response = await client.get(
                    path, headers={'X-Session-API-Key': keys['sb-first']}
                )
                second_response = await client.get(
                    path, headers={'X-Session-API-Key': keys['sb-second']}
                )
                assert first_response.json()['access_token'] == 'first-private-token'
                assert second_response.json()['access_token'] == 'second-private-token'
                assert (
                    'refresh-private' not in first_response.text + second_response.text
                )
                assert first_response.headers['cache-control'] == 'no-store'
                for sandbox_id in ('sb-team', 'sb-legacy', 'sb-old'):
                    denied = await client.get(
                        path, headers={'X-Session-API-Key': keys[sandbox_id]}
                    )
                    assert denied.status_code == 403
                    assert 'first-private-token' not in denied.text
                states['sb-first'] = 'paused'
                assert (
                    await client.get(
                        path, headers={'X-Session-API-Key': keys['sb-first']}
                    )
                ).status_code == 401
                states['sb-first'] = 'running'
                await OpenAISubscriptionService(store).logout(first.id)
                assert (
                    await client.get(
                        path, headers={'X-Session-API-Key': keys['sb-first']}
                    )
                ).status_code == 404
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(sdk_credentials, 'SubscriptionBrokerCredentialStore'),
    reason='requires broker-capable SDK',
)
async def test_login_to_real_sdk_broker_roundtrip(
    provider, create_user, create_org, async_session_maker, app_db_session, monkeypatch
):
    monkeypatch.setattr(
        server_config, 'user_auth_class', 'server.auth.saas_user_auth.SaasUserAuth'
    )
    user_id = uuid4()
    create_org(id=user_id)
    create_user(id=user_id, current_org_id=user_id)
    user_key = await ApiKeyStore().create_api_key(str(user_id), org_id=user_id)
    session_key = 'sandbox-private-key'
    async with async_session_maker() as session, session.begin():
        session.add(
            StoredSandbox(
                id='sb-live',
                backend='remote',
                created_by_user_id=str(user_id),
                workspace_id=str(user_id),
                subscription_broker_ready=True,
                sandbox_spec_id='default',
                session_api_key_hash=hash_session_api_key(session_key),
            )
        )

    async def runtime(_request):
        return web.json_response({'status': 'running', 'session_api_key': session_key})

    runtime_app = web.Application()
    runtime_app.router.add_get('/sessions/{sandbox_id}', runtime)
    runner = web.AppRunner(runtime_app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    app = FastAPI()
    app.include_router(router)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen()
    address = f'http://127.0.0.1:{sock.getsockname()[1]}'
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', lifespan='off'))
    task = None
    try:
        async with httpx.AsyncClient() as runtime_client:

            @asynccontextmanager
            async def sandbox_service(_state):
                async with async_session_maker() as session:
                    yield RemoteSandboxService(
                        sandbox_spec_service=None,
                        api_url=f'http://127.0.0.1:{port}',
                        api_key='runtime-key',
                        web_url=None,
                        resource_factor=1,
                        runtime_class=None,
                        start_sandbox_timeout=10,
                        max_num_sandboxes=1,
                        user_context=ADMIN,
                        httpx_client=runtime_client,
                        db_session=session,
                    )

            monkeypatch.setattr(session_auth, 'get_sandbox_service', sandbox_service)
            task = asyncio.create_task(server.serve(sockets=[sock]))
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError('Test server did not start')
                    await asyncio.sleep(0.01)
            async with httpx.AsyncClient(base_url=address) as client:
                headers = {'X-Access-Token': user_key}
                start = await client.post(
                    '/api/llm/subscription/openai/device/start', headers=headers
                )
                assert start.status_code == 200
                provider['ready'] = True
                poll = await client.post(
                    '/api/llm/subscription/openai/device/poll',
                    headers=headers,
                    json={'device_code': start.json()['device_code']},
                )
                assert poll.status_code == 200 and poll.json()['connected']
                broker = sdk_credentials.SubscriptionBrokerCredentialStore(
                    f'{address}/api/llm/subscription/openai/runtime/token', session_key
                )
                credential = await broker.aget('openai')
                assert credential.access_token == 'access-2'
                assert credential.refresh_token == ''
                assert (
                    await asyncio.to_thread(broker.get, 'openai')
                ).access_token == 'access-2'
                assert provider['refreshes'] == 1
                assert (
                    await client.post(
                        '/api/llm/subscription/openai/logout', headers=headers
                    )
                ).status_code == 200
                assert await broker.aget('openai') is None
    finally:
        server.should_exit = True
        if task is not None:
            await task
        sock.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_remote_sandbox_workspace_switch_and_owner_quota(
    create_user, create_org, async_session_maker, app_db_session, monkeypatch
):
    monkeypatch.setenv('OH_ENABLE_CLOUD_OPENAI_SUBSCRIPTION', 'true')
    user_id = uuid4()
    create_org(id=user_id)
    team = create_org()
    create_user(id=user_id, current_org_id=user_id)
    runtimes = {}
    starts = []

    async def start(request):
        body = await request.json()
        starts.append(body)
        sandbox_id = body['session_id']
        runtime = {
            'session_id': sandbox_id,
            'runtime_id': sandbox_id,
            'status': 'running',
            'session_api_key': f'key-{sandbox_id}',
        }
        runtimes[sandbox_id] = runtime
        return web.json_response(runtime)

    async def list_runtimes(_request):
        return web.json_response(
            {'runtimes': [r for r in runtimes.values() if r['status'] == 'running']}
        )

    async def get_runtime(request):
        return web.json_response(runtimes[request.match_info['sandbox_id']])

    async def batch(request):
        return web.json_response(
            [runtimes.get(sandbox_id) for sandbox_id in request.query.getall('ids', [])]
        )

    async def transition(request):
        sandbox_id = (await request.json())['runtime_id']
        runtime = runtimes[sandbox_id]
        runtime['status'] = 'paused' if request.path == '/pause' else 'running'
        if runtime['status'] == 'running':
            runtime['session_api_key'] = f'new-key-{sandbox_id}'
        return web.json_response(runtime)

    runtime_app = web.Application()
    runtime_app.router.add_post('/start', start)
    runtime_app.router.add_get('/list', list_runtimes)
    runtime_app.router.add_get('/sessions/batch', batch)
    runtime_app.router.add_get('/sessions/{sandbox_id}', get_runtime)
    runtime_app.router.add_post('/pause', transition)
    runtime_app.router.add_post('/resume', transition)
    runner = web.AppRunner(runtime_app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    specs = PresetSandboxSpecService(
        specs=[SandboxSpecInfo(id='test', command=['agent'])]
    )

    def context(org_id):
        return AuthUserContext(
            user_auth=SaasUserAuth(
                user_id=str(user_id), refresh_token=SecretStr(''), api_key_org_id=org_id
            ),
            _user_info=UserInfo(id=str(user_id)),
        )

    try:
        async with httpx.AsyncClient() as client, async_session_maker() as session:

            def service(org_id):
                return RemoteSandboxService(
                    sandbox_spec_service=specs,
                    api_url=f'http://127.0.0.1:{port}',
                    api_key='runtime-key',
                    web_url='https://app.example.com',
                    resource_factor=1,
                    runtime_class=None,
                    start_sandbox_timeout=10,
                    max_num_sandboxes=1,
                    user_context=context(org_id),
                    httpx_client=client,
                    db_session=session,
                )

            personal, team_service = service(user_id), service(team.id)
            with pytest.raises(AuthError, match='personal workspace'):
                await team_service.start_sandbox(
                    sandbox_id='forbidden', subscription_broker=True
                )
            assert starts == []
            first = await personal.start_sandbox(
                sandbox_id='personal', subscription_broker=True
            )
            assert first.workspace_id == str(user_id)
            assert 'workspace_id' not in first.model_dump()
            await session.flush()
            assert (await team_service.search_sandboxes()).items == []
            assert await team_service.get_sandbox('personal') is None
            assert await team_service.resume_sandbox('personal') is False
            second = await team_service.start_sandbox(sandbox_id='team')
            await session.flush()
            assert second.workspace_id == str(team.id)
            assert runtimes['personal']['status'] == 'paused'
            assert await personal.get_sandbox('team') is not None
            assert {item.id for item in (await personal.search_sandboxes()).items} == {
                'personal',
                'team',
            }
            assert (await team_service.search_sandboxes()).items[0].id == 'team'
            assert await personal.resume_sandbox('personal') is True
            assert runtimes['team']['status'] == 'paused'
            assert runtimes['personal']['status'] == 'running'
            broker_env, ordinary_env = [body['environment'] for body in starts]
            assert broker_env['OH_OPENAI_SUBSCRIPTION_MODE'] == 'broker'
            assert (
                broker_env['OH_OPENAI_SUBSCRIPTION_BROKER_URL']
                == 'https://app.example.com/api/llm/subscription/openai/runtime/token'
            )
            assert ordinary_env['OH_OPENAI_SUBSCRIPTION_MODE'] == 'disabled'
            assert 'OH_OPENAI_SUBSCRIPTION_BROKER_URL' not in ordinary_env
    finally:
        await runner.cleanup()
