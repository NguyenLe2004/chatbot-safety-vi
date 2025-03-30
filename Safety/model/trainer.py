from typing import Optional, Dict, Any
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from sklearn.metrics import jaccard_score, hamming_loss
import numpy as np
from model.Loss import FocalLoss
import evaluate
import torch
from torch.utils.data import Dataset


class DataAugmentationCallback(TrainerCallback):
    """Callback to trigger data augmentation at the end of each epoch"""
    
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset):
        """
        Args:
            train_dataset: Training dataset with augmentation capabilities
            val_dataset: Validation dataset with augmentation capabilities
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def on_epoch_end(self, args: TrainingArguments, state: Any, control: Any, **kwargs):
        """Trigger augmentation at epoch end"""
        self.train_dataset.start_augmentation()
        self.val_dataset.start_augmentation()


class BertTrainer(Trainer):
    """Custom Trainer implementation for BERT classification with enhanced metrics"""
    
    def __init__(
        self,
        model,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        args: Optional[TrainingArguments] = None,
        callbacks: Optional[list] = None,
        **kwargs
    ):
        """
        Initialize the custom trainer with focal loss and metrics
        
        Args:
            model: Pre-trained BERT model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            args: Training arguments
            callbacks: List of callbacks
            **kwargs: Additional arguments
        """
        self._focal_loss = FocalLoss()
        
        # Initialize evaluation metrics
        self.clf_metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])
        self.auc_score = evaluate.load("roc_auc")
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
            **kwargs
        )
        print(model.device)

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> torch.Tensor:
        """
        Compute focal loss for model training
        
        Args:
            model: The model to train
            inputs: Dictionary containing model inputs
            return_outputs: Whether to return model outputs
            
        Returns:
            The computed loss value
        """
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self._focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

    def compute_metrics(self, eval_pred: tuple) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            eval_pred: Tuple containing predictions and labels
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        
        # Convert logits to probabilities and binary predictions
        predictions_proba = torch.sigmoid(torch.tensor(predictions)).numpy()
        binary_preds = (predictions_proba > 0.5).astype(int).reshape(-1)
        labels = labels.astype(int).reshape(-1)
        
        # Compute classification metrics
        metrics_output = self.clf_metrics.compute(
            predictions=binary_preds,
            references=labels
        )
        
        # Compute ROC AUC
        roc_auc = self.auc_score.compute(
            references=labels,
            prediction_scores=predictions_proba.reshape(-1)
        )
        
        # Additional metrics
        return {
            **metrics_output,
            "hamming_loss": hamming_loss(binary_preds, labels),
            "jaccard_score": jaccard_score(binary_preds, labels, average='micro'),
            "roc_auc": roc_auc["roc_auc"],
        }


def setup_trainer(
    model,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    args: TrainingArguments
) -> BertTrainer:
    """
    Helper function to initialize and configure the custom trainer
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        args: Training arguments
        
    Returns:
        Configured BertTrainer instance
    """
    return BertTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[DataAugmentationCallback(train_dataset, eval_dataset)]
    )