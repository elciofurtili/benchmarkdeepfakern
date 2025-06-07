import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1_score": f1_score(labels, preds)
    }


def evaluate_models(models: dict, dataloader, device="cpu"):
    results = {}
    for name, model in models.items():
        labels, preds = evaluate_model(model, dataloader, device)
        metrics = compute_metrics(labels, preds)
        results[name] = metrics
    return pd.DataFrame(results).T


def save_metrics_to_csv(results_df, filename="view/results/metrics_results.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results_df.to_csv(filename)