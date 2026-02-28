from PIL import Image
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================
# Dataset
# =========================
class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = os.listdir(root)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# =========================
# Transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = FlatImageDataset("mini-mini-dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# =========================
# Model
# =========================
cnn = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V2
)
cnn.eval()


# =========================
# Metrics collection (image only)
# =========================
def get_metrics(images, cnn):
    """
    Collects only:
    - Inference latency
    - CNN confidence / entropy metrics
    """
    latencies = []
    confidences = []
    entropies = []
    top2_diffs = []

    for img in images:
        start = time.perf_counter()

        # ---- Inference ----
        with torch.no_grad():
            logits = cnn(img.unsqueeze(0))

        # ---- Latency ----
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        # ---- CNN output metrics ----
        probs = F.softmax(logits, dim=1)
        confidences.append(torch.max(probs).item())
        entropies.append(-(probs * torch.log(probs + 1e-8)).sum().item())

        top2 = torch.topk(probs, 2).values
        top2_diffs.append((top2[0, 0] - top2[0, 1]).item())

    # ---- Aggregate metrics ----
    metrics = {
        "latency_mean_ms": np.mean(latencies),
        "latency_std_ms": np.std(latencies),
        "confidence_mean": np.mean(confidences),
        "confidence_std": np.std(confidences),
        "entropy_mean": np.mean(entropies),
        "entropy_std": np.std(entropies),
        "top2_diff_mean": np.mean(top2_diffs),
    }

    return metrics


# =========================
# Run collection
# =========================
final_metrics = []

for images in dataloader:
    final_metrics.append(get_metrics(images, cnn))

final_metrics_df = pd.DataFrame(final_metrics)
final_metrics_df.to_csv("sub_cpu_stress_metrics.csv", index=False)

print("Metrics written to sub_normal_metrics.csv")