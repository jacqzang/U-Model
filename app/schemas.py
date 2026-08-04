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