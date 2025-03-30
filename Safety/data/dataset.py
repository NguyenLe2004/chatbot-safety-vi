from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from data.augmentation import get_augmentation_functions
import torch
import pandas as pd
from typing import Tuple, List, Optional, Callable

LABEL_COLUMNS = [
    "Criminal Planning/Confessions",
    "Fraud & Legal Violations",
    "Harassment",
    "Hate",
    "Offensive & Harmful Language",
    "Threat & Violence"
]


class SafetyDataset(torch.utils.data.Dataset):
    """Custom dataset for safety classification with text augmentation support"""
    
    def __init__(
        self,
        data: Dataset,
        tokenizer: Callable,
        augment_fns: Optional[List[Callable]] = None,
        sample: bool = True,
        sample_frac: float = 2/3,
        random_state: int = 42
    ):
        """
        Initialize the safety dataset
        
        Args:
            data: HuggingFace Dataset containing the raw data
            tokenizer: Text tokenizer function
            augment_fns: List of augmentation functions
            sample: Whether to perform sampling
            sample_frac: Fraction of data to sample
            random_state: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.augment_fns = augment_fns
        self.random_state = random_state
        self.sample_frac = sample_frac
        
        # Convert to pandas DataFrame and initialize tracking
        self.df = data.to_pandas().copy()
        self.df["selected"] = False 
        
        # Handle sampling
        self.df_sample = self._sample_data() if sample else self.df
        
        # Initialize tokenized dataset
        self.tokenized_data = Dataset.from_pandas(self.df_sample).map(
            self._preprocess_function,
            batched=True,
        )
        
        # Apply initial augmentation if needed
        if self.augment_fns:
            self.apply_augmentation()

    def _sample_data(self) -> pd.DataFrame:
        """Perform stratified sampling of the dataset"""
        if not self.df["selected"].any():
            # Initial sampling
            sample_df = self.df.sample(
                frac=self.sample_frac,
                random_state=self.random_state
            )
        else:
            # Subsequent sampling - combine new and previously selected samples
            new_samples = self.df[~self.df["selected"]]
            prev_samples = self.df[self.df["selected"]].sample(
                frac=1/3,
                random_state=self.random_state
            )
            sample_df = pd.concat([new_samples, prev_samples])
            
        # Update tracking and shuffle
        self.df.loc[sample_df.index, "selected"] = True
        return sample_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def _augment_text(self, text: str) -> str:
        """
        Apply augmentation to text with rate based on safety status
        
        Args:
            text: Input text to augment
            is_unsafe: Whether the text is unsafe
            
        Returns:
            Augmented text (or original if augmentation fails)
        """
        if not self.augment_fns:
            return text
        
        for augment_fn in self.augment_fns:
            try:
                augmented_text = augment_fn(text)
                if augmented_text and augmented_text.strip():
                    return augmented_text
            except Exception as e:
                pass
                
        return text

    def apply_augmentation(self):
        """Apply augmentation to the current dataset sample"""
        
        augmented_df = self.df_sample.copy()
        augmented_df["input"] = augmented_df.apply(
            lambda row: self._augment_text(
                row["input"]
            ),
            axis=1
        )
        
        # Re-tokenize augmented data
        self.tokenized_data = Dataset.from_pandas(augmented_df).map(
            self._preprocess_function,
            batched=True,
        )

    def _preprocess_function(self, batch: dict) -> dict:
        """
        Tokenize text and prepare model inputs
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Dictionary with tokenized inputs
        """
        tokenized = self.tokenizer(
            batch["input"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': batch["labels"]
        }

    def __len__(self) -> int:
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> dict:
        item = self.tokenized_data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids']),
            'attention_mask': torch.tensor(item['attention_mask']),
            'labels': torch.tensor(item['labels'], dtype=torch.float)
        }


def load_and_preprocess_data(
    data_path: str,
    tokenizer: Callable,
    augment_train: bool = True
) -> Tuple[SafetyDataset, SafetyDataset]:
    """
    Load and preprocess safety data
    
    Args:
        data_path: Path to dataset file
        tokenizer: Text tokenizer function
        augment_train: Whether to augment training data
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    df = load_safety_data(data_path)
    train_data, val_data = create_stratified_datasets(df)
    
    aug_fns = get_augmentation_functions(rate = 0.1) if augment_train else None
    
    train_dataset = SafetyDataset(
        train_data,
        tokenizer=tokenizer,
        augment_fns=aug_fns,
        sample=True
    )
    
    val_dataset = SafetyDataset(
        val_data,
        tokenizer=tokenizer,
        augment_fns=None,
        sample=False
    )
    
    return train_dataset, val_dataset


def load_safety_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess safety dataset
    
    Args:
        data_path: Path to dataset file
        
    Returns:
        Preprocessed DataFrame with labels and stratification column
    """
    dataset = load_dataset(data_path, data_files="ChatbotViSafety.csv", split="train")
    df = dataset.to_pandas()
    
    # Create multi-label target
    df["labels"] = df[LABEL_COLUMNS].values.tolist()
    
    return _prepare_stratification(df)


def _prepare_stratification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare stratification column by handling rare categories
    
    Args:
        df: Input DataFrame with categorical column
        
    Returns:
        DataFrame with added stratify_col
    """
    value_counts = df["categorical"].value_counts()
    rare_categories = value_counts[value_counts < 2].index.tolist()
    
    df["stratify_col"] = df["categorical"].apply(
        lambda cat: "less" if cat in rare_categories else cat
    )
    
    return df


def create_stratified_datasets(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[Dataset, Dataset]:
    """
    Create stratified train-test split datasets
    
    Args:
        df: Input DataFrame
        test_size: Fraction for validation set
        
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["stratify_col"],
        random_state=42
    )
    
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)