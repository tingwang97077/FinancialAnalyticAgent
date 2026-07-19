"""SQLAlchemy engine and session factories."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

type SessionFactory = sessionmaker[Session]


def create_rw_engine(database_url: str) -> Engine:
    """Create the read-write application and ingestion engine.

    Args:
        database_url: SQLAlchemy database URL for the application role.

    Returns:
        A pooled SQLAlchemy engine.

    Raises:
        ValueError: If the database URL is empty.
    """
    return _create_engine(database_url, read_only=False)


def create_readonly_engine(database_url: str) -> Engine:
    """Create the engine reserved for deterministic SQL queries.

    The database role remains the primary access-control boundary. The connection
    option adds a second read-only guard at the PostgreSQL session level.

    Args:
        database_url: SQLAlchemy database URL for the read-only role.

    Returns:
        A pooled SQLAlchemy engine whose transactions are read-only by default.

    Raises:
        ValueError: If the database URL is empty.
    """
    return _create_engine(database_url, read_only=True)


def create_session_factory(engine: Engine) -> SessionFactory:
    """Create a reusable SQLAlchemy session factory.

    Args:
        engine: Engine that will own database connections.

    Returns:
        A session factory bound to the supplied engine.
    """
    return sessionmaker(bind=engine, class_=Session, autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(session_factory: SessionFactory) -> Iterator[Session]:
    """Provide a session that always closes and rolls back on failure.

    Transaction commits remain explicit at call sites so read paths cannot commit
    accidentally and ingestion code controls its own transaction boundaries.

    Args:
        session_factory: Factory used to open the session.

    Yields:
        An open SQLAlchemy session.
    """
    session = session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _create_engine(database_url: str, *, read_only: bool) -> Engine:
    """Build an engine with common pool and connection safety settings."""
    normalized_url = database_url.strip()
    if not normalized_url:
        raise ValueError("Database URL must not be empty.")

    connect_args = {"options": "-c default_transaction_read_only=on"} if read_only else {}
    return create_engine(
        normalized_url,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
