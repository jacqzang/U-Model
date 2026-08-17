"""
Loads a dataset's images from S3 into a PyTorch-ready format: resized,
normalized, and labeled, split into training and testing sets.
"""
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from app.storage import s3_client, BUCKET_NAME

IMAGE_SIZE = 64  # all images get resized to 64x64 before entering the CNN

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # converts image to a PyTorch tensor, scales pixels 0-1
])


class S3ImageDataset(Dataset):
    """
    A PyTorch Dataset that pulls images directly from S3 on demand, rather
    than downloading everything to disk first.
    """
    def __init__(self, s3_prefix: str, class_names: list[str]):
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.samples = []  # list of (s3_key, label_index) tuples

        for class_name in class_names:
            prefix = f"{s3_prefix}/{class_name}/"
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
                for obj in page.get("Contents", []):
                    self.samples.append((obj["Key"], self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s3_key, label = self.samples[idx]
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=s3_key)
        image_bytes = response["Body"].read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image)
        return image_tensor, label