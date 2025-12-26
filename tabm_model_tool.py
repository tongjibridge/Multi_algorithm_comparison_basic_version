"""
TabM Model Wrapper
A convenient wrapper for the TabM model from Yandex Research.

This module provides two main classes:
- tabm_config: Configuration class for model hyperparameters
- tabm_model: Model wrapper with train and predict methods

Example usage:
    # For regression
    config = tabm_config(task_type='regression')
    model = tabm_model(config)
    model.fit(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_test)

    # For classification
    config = tabm_config(task_type='multiclass', n_classes=3)
    model = tabm_model(config)
    model.fit(X_train, y_train, X_val, y_val)
    predictions = model.predict(X_test)
"""

import math
from copy import deepcopy
from typing import Any, Literal, NamedTuple, Optional, Union

import numpy as np
import rtdl_num_embeddings
import scipy.special
import sklearn.preprocessing
import tabm
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor
import pandas as pd


class RegressionLabelStats(NamedTuple):
    """Statistics for regression label normalization."""

    mean: float
    std: float


class tabm_config:
    """
    Configuration class for TabM model.

    Parameters
    ----------
    task_type : {'regression', 'binclass', 'multiclass'}
        Type of machine learning task
    n_classes : int, optional
        Number of classes for classification tasks (required for multiclass)
    n_num_features : int
        Number of numerical features
    cat_cardinalities : list of int, optional
        Cardinalities of categorical features (e.g., [4, 7] for two categorical features)
    embedding_type : {'none', 'linear', 'periodic', 'piecewise'}, default='piecewise'
        Type of numerical feature embeddings
    n_bins : int, default=48
        Number of bins for piecewise linear embeddings
    d_embedding : int, default=16
        Dimension of embeddings
    lr : float, default=2e-3
        Learning rate
    weight_decay : float, default=3e-4
        Weight decay for AdamW optimizer
    batch_size : int, default=256
        Batch size for training
    patience : int, default=16
        Early stopping patience (number of epochs without improvement)
    max_epochs : int, default=1000000000
        Maximum number of training epochs
    gradient_clipping_norm : float or None, default=1.0
        Gradient clipping norm value
    share_training_batches : bool, default=True
        Whether MLPs share the same batches (True) or use different batches (False)
    amp_enabled : bool, default=False
        Whether to enable automatic mixed precision
    compile_model : bool, default=False
        Whether to use torch.compile
    device : str, default='cuda'
        Device to use ('cuda' or 'cpu')
    verbose : bool, default=True
        Whether to print training progress
    """

    def __init__(
        self,
        task_type: Literal["regression", "binclass", "multiclass"] = "regression",
        n_classes: Optional[int] = None,
        n_num_features: int = 8,
        cat_cardinalities: Optional[list[int]] = None,
        embedding_type: Literal[
            "none", "linear", "periodic", "piecewise"
        ] = "piecewise",
        n_bins: int = 48,
        d_embedding: int = 16,
        lr: float = 2e-3,
        weight_decay: float = 3e-4,
        batch_size: int = 256,
        patience: int = 16,
        max_epochs: int = 1_000_000_000,
        gradient_clipping_norm: Optional[float] = 1.0,
        share_training_batches: bool = True,
        amp_enabled: bool = False,
        compile_model: bool = False,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.task_type = task_type
        self.n_classes = n_classes
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities if cat_cardinalities else []
        self.embedding_type = embedding_type
        self.n_bins = n_bins
        self.d_embedding = d_embedding
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.max_epochs = max_epochs
        self.gradient_clipping_norm = gradient_clipping_norm
        self.share_training_batches = share_training_batches
        self.amp_enabled = amp_enabled
        self.compile_model = compile_model
        self.device = device
        self.verbose = verbose

        # Validation
        if task_type == "multiclass" and n_classes is None:
            raise ValueError("n_classes must be specified for multiclass tasks")
        if task_type != "regression" and n_classes is not None and n_classes < 2:
            raise ValueError("n_classes must be >= 2 for classification tasks")


class Tabm_model:
    """
    TabM model wrapper with train and predict methods.

    Parameters
    ----------
    config : tabm_config
        Configuration object containing model hyperparameters

    Attributes
    ----------
    model : tabm.TabM
        The underlying TabM model
    config : tabm_config
        Configuration object
    preprocessing : sklearn.preprocessing.QuantileTransformer or StandardScaler
        Fitted preprocessing for numerical features
    regression_label_stats : RegressionLabelStats or None
        Statistics for regression label normalization
    best_checkpoint : dict
        Best model checkpoint during training
    """

    def __init__(
        self,
        config: tabm_config,
    ):
        self.config = config
        self.model = None
        self.optimizer = None
        self.preprocessing = None
        self.regression_label_stats = None
        self.best_checkpoint = None
        self.device = None
        self.amp_dtype = None
        self.grad_scaler = None
        self.evaluation_mode = None
        self._is_fitted = False

        self.train_y = None
        self.val_y = None
        self.train_x_cat = None
        self.val_x_cat = None
        self.train_x_num = None
        self.val_x_num = None

        # Setup device
        self._setup_device()

        # Setup AMP
        self._setup_amp()

        # Setup evaluation mode
        self.evaluation_mode = (
            torch.no_grad if self.config.compile_model else torch.inference_mode
        )

    def _setup_device(self):
        """Setup device for training and inference."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        else:
            self.device = torch.device("cpu")

    def _setup_amp(self):
        """Setup automatic mixed precision."""
        if self.config.amp_enabled and torch.cuda.is_available():
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

            if self.amp_dtype == torch.float16:
                self.grad_scaler = torch.cuda.amp.GradScaler()
            else:
                self.grad_scaler = None
        else:
            self.amp_dtype = None

            self.grad_scaler = None

    def _create_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_cat_train: Optional[np.ndarray] = None,
        X_cat_val: Optional[np.ndarray] = None,
    ):
        """
        Create the TabM model based on configuration.
        Parameters
        ----------
        X_train : np.ndarray
            Training numerical features (n_samples, n_features)
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray, optional
            Validation numerical features. If None, uses part of training data
        y_val : np.ndarray, optional
            Validation labels
        X_cat_train : np.ndarray, optional
            Training categorical features
        X_cat_val : np.ndarray, optional
            Validation categorical features
        """
        # Create numerical embeddings
        # Validate inputs
        if X_train.shape[1] != self.config.n_num_features:
            raise ValueError(
                f"Expected {self.config.n_num_features} features, got {X_train.shape[1]}"
            )

        # If no validation set provided, split training data
        if X_val is None:
            val_size = min(int(len(X_train) * 0.2), 5000)
            indices = np.random.permutation(len(X_train))
            train_idx = indices[val_size:]
            val_idx = indices[:val_size]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            X_train = X_train[train_idx]
            y_train = y_train[train_idx]
            if X_cat_train is not None:
                X_cat_val = X_cat_train[val_idx]
                X_cat_train = X_cat_train[train_idx]

        # Store training data for embedding creation
        self._train_x_num = torch.as_tensor(X_train, device=self.device)

        # Preprocess features
        self.train_x_num = self._preprocess_features(X_train, fit=False)
        self.val_x_num = self._preprocess_features(X_val, fit=False)

        # Preprocess labels
        self.train_y = self._preprocess_labels(y_train, fit=False)
        self.val_y = self._preprocess_labels(y_val, fit=False)

        # Handle categorical features
        self.train_x_cat = (
            torch.as_tensor(X_cat_train, device=self.device)
            if X_cat_train is not None
            else None
        )
        self.val_x_cat = (
            torch.as_tensor(X_cat_val, device=self.device)
            if X_cat_val is not None
            else None
        )

        # Convert regression labels to float
        if self.config.task_type == "regression":
            self.train_y = self.train_y.float()
            self.val_y = self.val_y.float()

        num_embeddings = self._create_num_embeddings()

        # Determine output dimension
        d_out = 1 if self.config.n_classes is None else self.config.n_classes

        # Create model
        self.model = tabm.TabM.make(
            n_num_features=self.config.n_num_features,
            cat_cardinalities=self.config.cat_cardinalities,
            d_out=d_out,
            num_embeddings=num_embeddings,
        ).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Compile model if requested
        if self.config.compile_model:
            self.model = torch.compile(self.model)

    def _create_num_embeddings(self):
        """Create numerical feature embeddings based on configuration."""
        if self.config.embedding_type == "none":
            return None
        elif self.config.embedding_type == "linear":
            return rtdl_num_embeddings.LinearReLUEmbeddings(self.config.n_num_features)
        elif self.config.embedding_type == "periodic":
            return rtdl_num_embeddings.PeriodicEmbeddings(
                self.config.n_num_features, lite=False
            )
        elif self.config.embedding_type == "piecewise":
            return rtdl_num_embeddings.PiecewiseLinearEmbeddings(
                rtdl_num_embeddings.compute_bins(
                    self._train_x_num, n_bins=self.config.n_bins
                ),
                d_embedding=self.config.d_embedding,
                activation=False,
                version="B",
            )
        else:
            raise ValueError(f"Unknown embedding type: {self.config.embedding_type}")

    def _preprocess_features(self, X_num: np.ndarray, fit: bool = True) -> Tensor:
        """Preprocess numerical features."""
        X_num = X_num.astype(np.float32)
        if fit:
            # Advanced preprocessing with noise
            return torch.as_tensor(
                self.preprocessing.transform(X_num), device=self.device
            )
        else:
            return torch.as_tensor(X_num, device=self.device)

    def _preprocess_labels(
        self, y: np.ndarray, normalize: bool = True, fit: bool = True
    ) -> Union[Tensor, tuple[Tensor, RegressionLabelStats]]:
        """Preprocess labels."""
        if self.config.task_type == "regression":
            y = y.astype(np.float32)

            if fit:
                if normalize:
                    self.regression_label_stats = RegressionLabelStats(
                        y.mean().item(), y.std().item()
                    )
                else:
                    self.regression_label_stats = RegressionLabelStats(0.0, 1.0)
                y_normalized = (
                    y - self.regression_label_stats.mean
                ) / self.regression_label_stats.std
                return torch.as_tensor(y_normalized, device=self.device).float()
            else:
                self.regression_label_stats = RegressionLabelStats(0.0, 1.0)
                return torch.as_tensor(y, device=self.device).float()
        else:
            y = y.astype(np.int64)
            return torch.as_tensor(y, device=self.device)

    def _make_checkpoint(self) -> dict[str, Any]:
        """Create a checkpoint of the current model state."""
        return deepcopy(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": getattr(self, "_current_epoch", -1),
                "metrics": getattr(
                    self, "_metrics", {"val": -math.inf, "test": -math.inf}
                ),
            }
        )

    def _apply_model(self, x_num: Tensor, x_cat: Optional[Tensor]) -> Tensor:
        """Apply model to input data."""
        return self.model(x_num, x_cat).squeeze(-1).float()

    def _loss_fn(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute loss."""
        y_pred = y_pred.flatten(0, 1)

        if self.config.share_training_batches:
            y_true = y_true.repeat_interleave(self.model.backbone.k)
        else:
            y_true = y_true.flatten(0, 1)

        if self.config.task_type == "regression":
            return nn.functional.mse_loss(y_pred, y_true)
        else:
            return nn.functional.cross_entropy(y_pred, y_true)

    def _evaluate(self, x_num: Tensor, x_cat: Optional[Tensor], y: Tensor) -> float:
        """Evaluate model on given data."""
        self.model.eval()

        eval_batch_size = 8096
        y_pred_list = []

        with self.evaluation_mode():
            for idx in torch.arange(len(y), device=self.device).split(eval_batch_size):
                y_pred_list.append(
                    self._apply_model(
                        x_num[idx], x_cat[idx] if x_cat is not None else None
                    )
                )

        y_pred = torch.cat(y_pred_list).cpu().numpy()

        if self.config.task_type == "regression":
            # Transform predictions back to original space
            assert self.regression_label_stats is not None
            y_pred = (
                y_pred * self.regression_label_stats.std
                + self.regression_label_stats.mean
            )
        else:
            # For classification, compute mean in probability space
            y_pred = scipy.special.softmax(y_pred, axis=-1)

        # Average over k predictions
        y_pred = y_pred.mean(1)
        y_true = y.cpu().numpy()

        # Compute score
        if self.config.task_type == "regression":
            score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
        else:
            score = sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))

        return float(score)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_cat_train: Optional[np.ndarray] = None,
        X_cat_val: Optional[np.ndarray] = None,
    ):
        """
        Train the TabM model.


        """
        self._create_model(
            X_train,
            y_train,
            X_val,
            y_val,
            X_cat_train,
            X_cat_val,
        )
        # Training loop
        train_size = len(self.train_y)
        batch_size = self.config.batch_size

        self._current_epoch = -1
        self._metrics = {"val": -math.inf}
        self.best_checkpoint = self._make_checkpoint()
        remaining_patience = self.config.patience

        for epoch in range(self.config.max_epochs):
            self._current_epoch = epoch

            # Create batches
            if self.config.share_training_batches:
                batches = torch.randperm(train_size, device=self.device).split(
                    batch_size
                )
            else:
                batches = (
                    torch.rand((train_size, self.model.backbone.k), device=self.device)
                    .argsort(dim=0)
                    .split(batch_size, dim=0)
                )

            # Training iterations
            for batch_idx in batches:
                self.model.train()
                self.optimizer.zero_grad()

                loss = self._loss_fn(
                    self._apply_model(
                        self.train_x_num[batch_idx],
                        self.train_x_cat[batch_idx]
                        if self.train_x_cat is not None
                        else None,
                    ),
                    self.train_y[batch_idx],
                )

                # Backward pass
                if self.grad_scaler is None:
                    loss.backward()
                else:
                    self.grad_scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clipping_norm is not None:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clipping_norm
                    )

                # Optimizer step
                if self.grad_scaler is None:
                    self.optimizer.step()
                else:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()

            # Evaluation
            val_score = self._evaluate(self.val_x_num, self.val_x_cat, self.val_y)
            self._metrics["val"] = val_score

            val_score_improved = val_score > self.best_checkpoint["metrics"]["val"]

            if self.config.verbose:
                print(
                    f"{'*' if val_score_improved else ' '}"
                    f" [epoch] {epoch:<3}"
                    f" [val] {val_score:.3f}"
                )

            # Checkpoint management
            if val_score_improved:
                self.best_checkpoint = self._make_checkpoint()
                remaining_patience = self.config.patience
            else:
                remaining_patience -= 1

            # Early stopping
            if remaining_patience < 0:
                if self.config.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Load best checkpoint
        self.model.load_state_dict(self.best_checkpoint["model"])
        self._is_fitted = True

        if self.config.verbose:
            print("\n[Summary]")
            print(f"best epoch:  {self.best_checkpoint['epoch']}")
            print(f"val score:   {self.best_checkpoint['metrics']['val']:.4f}")

        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        X_cat: Optional[np.ndarray] = None,
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            Numerical features (n_samples, n_features)
        X_cat : np.ndarray, optional
            Categorical features
        return_proba : bool, default=False
            For classification, return class probabilities (only for classification tasks)

        Returns
        -------
        np.ndarray
            Predictions. For regression: continuous values.
            For classification with return_proba=False: class labels.
            For classification with return_proba=True: class probabilities.
        """
        # 如果Xs是DataFrame，转换为numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Preprocess features
        x_num = self._preprocess_features(X, fit=False)

        # Handle categorical features
        x_cat = (
            torch.as_tensor(X_cat, device=self.device) if X_cat is not None else None
        )

        # Prediction
        self.model.eval()
        eval_batch_size = 8096
        y_pred_list = []

        with self.evaluation_mode():
            for idx in torch.arange(len(x_num), device=self.device).split(
                eval_batch_size
            ):
                res = self._apply_model(
                    x_num[idx], x_cat[idx] if x_cat is not None else None
                )
                y_pred_list.append(res)

        y_pred = torch.cat(y_pred_list).cpu().numpy()

        if self.config.task_type == "regression":
            # Transform predictions back to original space
            assert self.regression_label_stats is not None
            y_pred = (
                y_pred * self.regression_label_stats.std
                + self.regression_label_stats.mean
            )
            # Average over k predictions
            y_pred = y_pred.mean(1)
            return y_pred
        else:
            # For classification, compute mean in probability space
            y_pred = scipy.special.softmax(y_pred, axis=-1)
            y_pred = y_pred.mean(1)

            if return_proba:
                return y_pred
            else:
                return y_pred.argmax(1)

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint to file.

        Parameters
        ----------
        path : str
            Path to save the checkpoint
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before saving")

        checkpoint = {
            "model": self.model,
            "config": self.config,
            "preprocessing": self.preprocessing,
            "regression_label_stats": self.regression_label_stats,
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        Load model from checkpoint.

        Parameters
        ----------
        path : str
            Path to the checkpoint file

        Returns
        -------
        tabm_model
            Loaded model instance
        """
        checkpoint = torch.load(path)
        self.model = checkpoint["model"]
        self.config = checkpoint["config"]
        self.preprocessing = checkpoint["preprocessing"]
        self.regression_label_stats = checkpoint["regression_label_stats"]
        self._is_fitted = True
        self.model.to(self.device)


# device_type = "cuda" if torch.cuda.is_available() else "cpu"
# amp_enabled = False
# amp_dtype = (
#     (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
#     if amp_enabled
#     else None
# )

# if __name__ == "__main__":
#     # Example usage
#     import sklearn.datasets
#     import sklearn.model_selection

#     print("=" * 60)
#     print("TabM Model Wrapper - Example Usage")
#     print("=" * 60)

#     # Load dataset
#     dataset = pd.read_excel(r"./公众号Lvy的口袋数据集.xlsx")
#     X = dataset.drop(columns=["PV_output"]).astype(np.float32)
#     y = dataset["PV_output"].astype(np.float32)

#     # Split data
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     print("\n[Regression Task]")
#     print(f"Training samples: {len(X_train)}")
#     print(f"Test samples: {len(X_test)}")
#     print(f"Features: {X.shape[1]}")

#     # Create config
#     config = tabm_config(
#         task_type="regression",
#         n_num_features=X.shape[1],
#         embedding_type="piecewise",
#         batch_size=256,
#         patience=15,
#         max_epochs=100,
#         verbose=True,
#         device="cuda",
#     )

#     # Create and train model
#     model = Tabm_model(config)
#     model.fit(X_train.values, y_train.values)

#     # Make predictions
#     predictions = model.predict(X_test.values)
#     MSE = sklearn.metrics.mean_squared_error(y_test, predictions)
#     rmse = np.sqrt(MSE)
#     R2 = sklearn.metrics.r2_score(y_test, predictions)
#     print(f"\nTest RMSE: {rmse:.4f} MSE: {MSE:.4f} R2: {R2:.4f}")
#     print("=" * 60)
