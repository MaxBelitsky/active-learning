import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def evaluate(model, data_loader, head_only=False):
    """
    Evaluate the model on the given data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): A PyTorch DataLoader containing the evaluation dataset.
        head_only (bool): Whether to evaluate only the classifier head.

    Returns:
        dict: The evaluation results (e.g., accuracy, loss).
    """
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Ensure we use the same device as the model

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()  # Loss function for evaluation

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in tqdm(data_loader):
            if head_only:
                inputs = batch["embeddings"].to(device)
                print("Happens")
            else:
                inputs = batch["pixel_values"].to(device)
                print("Happens 2")
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(inputs)

            if head_only:
                logits = outputs[:, 0, :]
            else:
                outputs = outputs.logits

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Get predictions
            _, predictions = torch.max(logits, dim=1)

            # Accumulate predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(data_loader)

    # Prepare results
    results = {
        "accuracy": accuracy,
        "loss": avg_loss
    }

    return results
