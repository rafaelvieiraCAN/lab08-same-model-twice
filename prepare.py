"""Prepare the bike rentals data for PyTorch."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]

CATEGORICAL = [
    "station_type",
]


def make_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and encode categories."""

    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
        ]
    )


class BikeDataset(Dataset):
    """Stores features and targets as float32 tensors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass
class Splits:
    train: BikeDataset
    val: BikeDataset
    test: BikeDataset
    preprocessor: ColumnTransformer

    @property
    def n_features(self) -> int:
        return self.train.x.shape[1]


def make_splits(seed: int = 42) -> Splits:
    """Create train, validation and test datasets."""

    frame = make_rentals(seed=seed)

    train_df, rest_df = train_test_split(
        frame,
        test_size=0.4,
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        rest_df,
        test_size=0.5,
        random_state=seed,
    )

    preprocessor = make_preprocessor()

    preprocessor.fit(train_df[NUMERIC + CATEGORICAL])

    def build(df):
        features = preprocessor.transform(df[NUMERIC + CATEGORICAL])

        labels = df[TARGET].to_numpy()

        return BikeDataset(
            features,
            labels,
        )

    return Splits(
        train=build(train_df),
        val=build(val_df),
        test=build(test_df),
        preprocessor=preprocessor,
    )


def make_loaders(
    splits: Splits,
    batch_size: int = 64,
    seed: int = 42,
):
    """Create DataLoaders."""

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        splits.val,
        batch_size=256,
    )

    test_loader = DataLoader(
        splits.test,
        batch_size=256,
    )

    return train_loader, val_loader, test_loader
