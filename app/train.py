"""
Trains a CNN on a dataset and returns the results: final accuracy and a
confusion matrix, matching what the spec's Results page needs.
"""
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score

from app.cnn import SimpleCNN
from app.dataset_loader import S3ImageDataset


def train_model(
    s3_prefix: str,
    class_names: list[str],
    epochs: int = 20,
    model_size: str = "medium",
    train_test_split: float = 0.8,
) -> dict:
    """
    Trains synchronously (blocks until done) — no live progress yet, that's
    Phase 2. Returns final accuracy and confusion matrix.
    """
    full_dataset = S3ImageDataset(s3_prefix=s3_prefix, class_names=class_names)

    train_size = int(len(full_dataset) * train_test_split)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = SimpleCNN(num_classes=len(class_names), model_size=model_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()

    # --- Training loop ---
    model.train()  # puts the model in "learning mode" (activates dropout, etc.)
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()          # clear gradients from the last batch
            outputs = model(images)        # run images through the CNN
            loss = loss_function(outputs, labels)  # how wrong were we?
            loss.backward()                # figure out how to adjust to be less wrong
            optimizer.step()               # actually adjust the model's weights

    # --- Evaluation on held-out test images ---
    model.eval()  # switches to "prediction mode" (deactivates dropout)
    all_preds = []
    all_labels = []
    with torch.no_grad():  # no need to track gradients when just evaluating
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)  # pick the highest-scoring class
            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "final_accuracy": accuracy,
        "confusion_matrix": cm,
        "class_names": class_names,
    }