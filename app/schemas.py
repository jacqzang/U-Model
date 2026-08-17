from pydantic import BaseModel, EmailStr


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config:
        from_attributes = True  # lets this read directly from a SQLAlchemy object


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TrainRequest(BaseModel):
    dataset_id: int
    epochs: int = 20
    model_size: str = "medium"
    train_test_split: float = 0.8
    confidence_threshold: float = 0.70