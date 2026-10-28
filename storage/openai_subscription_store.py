from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from storage.base import Base
from storage.database import a_session_maker
from storage.encrypt_utils import EncryptedJSON
from storage.user import User


class OpenAISubscription(Base):
    __tablename__ = 'openai_subscription'

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey('user.id', ondelete='CASCADE'), primary_key=True
    )
    credentials: Mapped[dict[str, Any] | None] = mapped_column(
        EncryptedJSON, nullable=True
    )
    pending_login: Mapped[dict[str, Any] | None] = mapped_column(
        EncryptedJSON, nullable=True
    )


class OpenAISubscriptionStore:
    def __init__(
        self,
        session_factory: Callable[
            [], AbstractAsyncContextManager[AsyncSession]
        ] = a_session_maker,
    ) -> None:
        self._session_factory = session_factory

    @asynccontextmanager
    async def locked(self, user_id: UUID) -> AsyncIterator[OpenAISubscription]:
        async with self._session_factory() as session, session.begin():
            # Lock the user row even before a subscription row exists, so first
            # login, token refresh and logout all serialize for the same user.
            user = await session.scalar(
                select(User.id).where(User.id == user_id).with_for_update()
            )
            if user is None:
                raise LookupError('User not found')
            state = await session.get(OpenAISubscription, user_id)
            if state is None:
                state = OpenAISubscription(user_id=user_id)
                session.add(state)
            yield state
