"""
Handles uploading dataset images to S3.
"""
import os
from dotenv import load_dotenv
import boto3

load_dotenv()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")


def upload_dataset_images(zip_path: str, user_id: int, dataset_name: str, image_paths: list) -> str:
    """
    Extracts images from the zip and uploads each one to S3, under a path like:
    datasets/{user_id}/{dataset_name}/{class_name}/{filename}

    Returns the S3 prefix (folder path) where this dataset now lives.
    """
    import zipfile
    from pathlib import Path

    s3_prefix = f"datasets/{user_id}/{dataset_name}"

    with zipfile.ZipFile(zip_path, "r") as zf:
        for path in image_paths:
            # image_paths are the *stripped* relative paths, e.g. "cats/cat1.jpg"
            # but zf.open() needs the *original* name inside the zip, which may
            # still have the wrapper folder. Try both.
            original_name = str(path)
            try:
                file_bytes = zf.read(original_name)
            except KeyError:
                # search for a matching entry (handles the stripped-wrapper case)
                match = next((n for n in zf.namelist() if n.endswith(original_name)), None)
                if not match:
                    continue
                file_bytes = zf.read(match)

            s3_key = f"{s3_prefix}/{path}"
            s3_client.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=file_bytes)

    return s3_prefix