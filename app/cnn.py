"""
The CNN architecture used for all training jobs. "Model size" (small/medium/large)
just scales the number of filters in each conv layer — same architecture, different
capacity. 
"""
import torch
import torch.nn as nn

# Maps the user-facing "model size" choice to actual filter counts per layer.
MODEL_SIZE_FILTERS = {
    "small": (32, 64, 128),
    "medium": (64, 128, 256),
    "large": (128, 256, 512),
}

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, model_size: str = "medium"):
        super().__init__()
        f1, f2, f3 = MODEL_SIZE_FILTERS[model_size]

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # halves height & width

            # Block 2
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(f2, f3, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(f3 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # flatten everything except the batch dimension
        x = self.dropout(x)
        x = self.classifier(x)
        return x  # raw logits — softmax gets applied later, not inside the model