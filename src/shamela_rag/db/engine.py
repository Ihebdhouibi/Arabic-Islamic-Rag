"""Database engine and session factory."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from shamela_rag.config import get_settings


def get_engine() -> Engine:
    return create_engine(get_settings().sqlalchemy_dsn, future=True)


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine if engine is not None else get_engine(),
        autoflush=False,
        expire_on_commit=False,
    )
