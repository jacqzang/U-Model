"""
ORM models for the 5 tables: users, datasets, training_requests, training_logs, models.
Each class here maps to a Postgres table.
"""
from datetime import datetime, timezone
from sqlalchemy import String, Integer, ForeignKey, DateTime, JSON, Float, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    datasets: Mapped[list["Dataset"]] = relationship(back_populates="owner")
    training_requests: Mapped[list["TrainingRequest"]] = relationship(back_populates="owner")
    models: Mapped[list["MLModel"]] = relationship(back_populates="owner")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    class_names: Mapped[list] = mapped_column(JSON, nullable=False)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    s3_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    owner: Mapped["User"] = relationship(back_populates="datasets")
    training_requests: Mapped[list["TrainingRequest"]] = relationship(back_populates="dataset")


class TrainingRequest(Base):
    __tablename__ = "training_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    status: Mapped[str] = mapped_column(String, default="queued")
    epochs: Mapped[int] = mapped_column(Integer, default=20)
    model_size: Mapped[str] = mapped_column(String, default="medium")
    train_test_split: Mapped[float] = mapped_column(Float, default=0.8)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.70)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    owner: Mapped["User"] = relationship(back_populates="training_requests")
    dataset: Mapped["Dataset"] = relationship(back_populates="training_requests")
    logs: Mapped[list["TrainingLog"]] = relationship(back_populates="request")
    model: Mapped["MLModel"] = relationship(back_populates="request", uselist=False)


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("training_requests.id"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    train_loss: Mapped[float] = mapped_column(Float, nullable=False)
    val_loss: Mapped[float] = mapped_column(Float, nullable=False)
    train_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    val_accuracy: Mapped[float] = mapped_column(Float, nullable=False)

    request: Mapped["TrainingRequest"] = relationship(back_populates="logs")


class MLModel(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("training_requests.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    s3_path: Mapped[str] = mapped_column(String, nullable=False)
    final_accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    confusion_matrix: Mapped[dict] = mapped_column(JSON, nullable=False)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.70)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    request: Mapped["TrainingRequest"] = relationship(back_populates="model")
    owner: Mapped["User"] = relationship(back_populates="models")