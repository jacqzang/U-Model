from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import engine, Base, get_db
from app import models, schemas, auth, validation, storage, train
from pathlib import Path

import shutil
import tempfile

app = FastAPI(title="U-Model API")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/health/db")
def health_check_db(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}


@app.post("/auth/register", response_model=schemas.UserOut)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.email == user_in.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = models.User(
        email=user_in.email,
        hashed_password=auth.hash_password(user_in.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/auth/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    token = auth.create_access_token(data={"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/auth/me", response_model=schemas.UserOut)
def read_current_user(current_user: models.User = Depends(auth.get_current_user)):
    return current_user


@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        report = validation.validate_dataset_zip(tmp_path)

        if not report["valid"]:
            return report  # don't upload anything if validation failed

        dataset_name = file.filename.replace(".zip", "")
        s3_prefix = storage.upload_dataset_images(
            zip_path=tmp_path,
            user_id=current_user.id,
            dataset_name=dataset_name,
            image_paths=report["image_paths"],
        )

        total_images = sum(report["classes"].values())
        dataset = models.Dataset(
            user_id=current_user.id,
            name=dataset_name,
            class_names=list(report["classes"].keys()),
            image_count=total_images,
            s3_path=s3_prefix,
        )
        db.add(dataset)
        db.commit()
        db.refresh(dataset)

        report["dataset_id"] = dataset.id
        report["s3_path"] = s3_prefix
        return report
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/train")
def start_training(
    train_in: schemas.TrainRequest,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    dataset = db.get(models.Dataset, train_in.dataset_id)
    if not dataset or dataset.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Dataset not found")

    request = models.TrainingRequest(
        user_id=current_user.id,
        dataset_id=dataset.id,
        status="running",
        epochs=train_in.epochs,
        model_size=train_in.model_size,
        train_test_split=train_in.train_test_split,
        confidence_threshold=train_in.confidence_threshold,
    )
    db.add(request)
    db.commit()
    db.refresh(request)

    result = train.train_model(
        s3_prefix=dataset.s3_path,
        class_names=dataset.class_names,
        epochs=train_in.epochs,
        model_size=train_in.model_size,
        train_test_split=train_in.train_test_split,
    )

    request.status = "complete"
    db.commit()

    model_record = models.MLModel(
        request_id=request.id,
        user_id=current_user.id,
        s3_path=f"models/{request.id}.pt",  # placeholder — actual .pt upload comes later
        final_accuracy=result["final_accuracy"],
        confusion_matrix=result["confusion_matrix"],
        confidence_threshold=train_in.confidence_threshold,
    )
    db.add(model_record)
    db.commit()
    db.refresh(model_record)

    return {
        "training_request_id": request.id,
        "model_id": model_record.id,
        "final_accuracy": result["final_accuracy"],
        "confusion_matrix": result["confusion_matrix"],
    }