from typing import Tuple, Callable
from datasets import Dataset, load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_preprocess_data(
    data_path: str,
    tokenizer: Callable,
    test_size : float,
) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess safety data
    
    Args:
        data_path: Path to dataset file
        tokenizer: Text tokenizer function
        augment_train: Whether to augment training data
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    train_dataset, val_dataset = load_data(data_path, test_size)
    train_dataset = train_dataset.map(lambda x: _preprocess_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: _preprocess_function(x, tokenizer), batched=True)
    return train_dataset, val_dataset

def load_data(data_path: str, test_size=0.2) -> Tuple[Dataset, Dataset]:
    """
    Load and preprocess safety dataset
    
    Args:
        data_path: Path to dataset file
        tokenizer: Tokenizer function
        test_size: Proportion of test set
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    dataset = load_dataset(data_path, data_files="ChatbotViSafety.csv", split="train")
    split_data = dataset.train_test_split(test_size=test_size)
    return split_data["train"], split_data["test"]


def _preprocess_function(ds, tokenizer):
    inputs = ds["input"]
    output = ds["output"]
    
    model_inputs = tokenizer(inputs, max_length=100, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=True)
    lm_labels = tokenizer(output, max_length=512, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=True)
    
    return {
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': lm_labels['input_ids'],
    }
