from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = 'runs'

    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, default='queued')
    prompt: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


engine = create_engine('sqlite:///data/runs.db')
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(engine)


def create_run(prompt: str, source: str, run_id: Optional[str] = None) -> Run:
    import uuid

    init_db()
    run = Run(id=run_id or uuid.uuid4().hex, prompt=prompt, source=source, status='queued')
    with SessionLocal() as session:
        session.add(run)
        session.commit()
        session.refresh(run)
    return run


def update_run(run_id: str, **fields) -> Run:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f'no such run: {run_id}')
        for key, value in fields.items():
            setattr(run, key, value)
        run.updated_at = _utcnow()
        session.commit()
        session.refresh(run)
    return run
