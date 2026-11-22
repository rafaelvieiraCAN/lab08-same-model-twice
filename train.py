"""Training loop for BikeNet."""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from model import BikeNet
from prepare import make_loaders, make_splits
from results import save_results

EPOCHS = 20
LEARNING_RATE = 0.01
SEED = 42


def run_epoch(
    model,
    loader,
    loss_fn,
    optimizer=None,
):
    """
    Run one epoch.

    If optimizer exists:
        train

    Otherwise:
        validate
    """

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            predictions = model(xb)

            loss = loss_fn(
                predictions,
                yb,
            )

            if training:
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            batch_size = len(yb)

            total_loss += loss.item() * batch_size

            mae = torch.mean(torch.abs(predictions - yb)).item()

            total_mae += mae * batch_size

            total_count += batch_size

    mean_loss = total_loss / total_count
    mean_mae = total_mae / total_count

    return mean_loss, mean_mae


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
):
    """Train model and restore best weights."""

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_mae = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
        )

        val_loss, val_mae = run_epoch(
            model,
            val_loader,
            loss_fn,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train MAE {train_mae:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val MAE {val_mae:.4f}"
        )

    model.load_state_dict(best_weights)

    print(f"\nBest epoch: {best_epoch} (Val Loss {best_val_loss:.4f})")

    return history, best_epoch


def save_learning_curve(
    history,
    best_epoch,
):
    """Save learning curve plot."""

    Path("reports").mkdir(
        exist_ok=True,
    )

    epochs = [row["epoch"] for row in history]

    train_loss = [row["train_loss"] for row in history]

    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
    )

    plt.axvline(
        best_epoch,
        color="red",
        linestyle="--",
        label=f"Best Epoch {best_epoch}",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig("reports/learning_curve.png")

    plt.close()


def main(seed=SEED):
    torch.manual_seed(seed)

    splits = make_splits(seed)

    train_loader, val_loader, test_loader = make_loaders(
        splits,
        seed=seed,
    )

    model = BikeNet(
        splits.n_features,
    )

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
    )

    save_learning_curve(
        history,
        best_epoch,
    )

    results = save_results(
        model,
        splits,
    )

    print("\nResults")
    print(results)

    print("\nTraining completed.")

    return {
        "history": history,
        "results": results,
    }


if __name__ == "__main__":
    main()
