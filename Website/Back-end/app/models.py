from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base

class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nim: Mapped[str] = mapped_column(String(80), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default='active', nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    templates = relationship('FaceTemplate', back_populates='user', cascade='all, delete-orphan')

class FaceTemplate(Base):
    __tablename__ = 'face_templates'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    embedding: Mapped[list] = mapped_column(JSON, nullable=False)  # 512 float values; keep access restricted
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    preprocessing_version: Mapped[str] = mapped_column(String(100), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), default='admin_reference_photo')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    user = relationship('User', back_populates='templates')
    __table_args__ = (Index('ix_face_template_version', 'model_version', 'preprocessing_version'),)

class VerificationEvent(Base):
    __tablename__ = 'verification_events'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    nim_masked: Mapped[str | None] = mapped_column(String(40), nullable=True)
    mode: Mapped[str] = mapped_column(String(10), default='1:1')
    face_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    detection_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_box_area_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold_used: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision_status: Mapped[str] = mapped_column(String(40), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    quality_warnings: Mapped[list] = mapped_column(JSON, default=list)
    quality_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    antispoof_status: Mapped[str] = mapped_column(String(40), default='not_configured')
    antispoof_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    detection_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    comparison_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    antispoof_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    preprocessing_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    threshold_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
