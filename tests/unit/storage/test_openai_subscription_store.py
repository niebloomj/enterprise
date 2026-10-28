import asyncio

import pytest
from sqlalchemy import delete, inspect, select, text

from storage.openai_subscription_store import (
    OpenAISubscription,
    OpenAISubscriptionStore,
)
from storage.user import User
from tests import postgres_testdb


def test_upgrade_and_downgrade_preserve_legacy_sandboxes(engine, test_database):
    postgres_testdb.run_alembic(
        test_database.server, test_database.name, 'downgrade', '204'
    )
    assert 'openai_subscription' not in inspect(engine).get_table_names()
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO v1_remote_sandbox (id, sandbox_spec_id) VALUES ('legacy', 'spec')"
            )
        )
    postgres_testdb.run_alembic(
        test_database.server, test_database.name, 'upgrade', 'head'
    )
    with engine.connect() as connection:
        legacy = connection.execute(
            text(
                "SELECT workspace_id, subscription_broker_ready FROM v1_remote_sandbox WHERE id='legacy'"
            )
        ).one()
    assert tuple(legacy) == (None, False)
    postgres_testdb.run_alembic(
        test_database.server, test_database.name, 'downgrade', '204'
    )
    assert 'openai_subscription' not in inspect(engine).get_table_names()
    assert 'workspace_id' not in {
        c['name'] for c in inspect(engine).get_columns('v1_remote_sandbox')
    }
    with engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT id FROM v1_remote_sandbox WHERE id='legacy'")
            ).scalar_one()
            == 'legacy'
        )


@pytest.mark.asyncio
async def test_credentials_are_encrypted_and_isolated_per_user(
    create_user, async_session_maker
):
    first = create_user()
    second = create_user()
    store = OpenAISubscriptionStore(async_session_maker)

    async with store.locked(first.id) as state:
        state.credentials = {
            'access_token': 'first-access',
            'refresh_token': 'first-refresh',
        }
        state.pending_login = {'device_auth_id': 'first-device'}
    async with store.locked(second.id) as state:
        assert state.credentials is None
        assert state.pending_login is None
        state.credentials = {
            'access_token': 'second-access',
            'refresh_token': 'second-refresh',
        }

    async with store.locked(first.id) as state:
        assert state.credentials['refresh_token'] == 'first-refresh'
        assert state.pending_login['device_auth_id'] == 'first-device'
    async with async_session_maker() as session:
        values = (
            await session.execute(
                text('SELECT credentials, pending_login FROM openai_subscription')
            )
        ).all()
    assert 'first-refresh' not in repr(values)
    assert 'first-device' not in repr(values)
    assert 'second-refresh' not in repr(values)


@pytest.mark.asyncio
async def test_same_user_operations_serialize_and_rollback(
    create_user, async_session_maker
):
    user = create_user()
    store = OpenAISubscriptionStore(async_session_maker)
    entered = asyncio.Event()

    async def update():
        async with store.locked(user.id) as state:
            entered.set()
            state.credentials = {'version': state.credentials['version'] + 1}

    async with store.locked(user.id) as state:
        state.credentials = {'version': 1}
        task = asyncio.create_task(update())
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(entered.wait(), timeout=0.1)
    await asyncio.wait_for(task, timeout=5)

    with pytest.raises(ValueError):
        async with store.locked(user.id) as state:
            state.credentials = {'version': 3}
            raise ValueError('rollback')
    async with store.locked(user.id) as state:
        assert state.credentials == {'version': 2}


@pytest.mark.asyncio
async def test_user_deletion_cascades_credentials(create_user, async_session_maker):
    user = create_user()
    store = OpenAISubscriptionStore(async_session_maker)
    async with store.locked(user.id) as state:
        state.credentials = {'refresh_token': 'sensitive'}
    async with async_session_maker() as session, session.begin():
        await session.execute(delete(User).where(User.id == user.id))
    async with async_session_maker() as session:
        assert await session.scalar(select(OpenAISubscription.user_id)) is None
    with pytest.raises(LookupError):
        async with store.locked(user.id):
            pass
