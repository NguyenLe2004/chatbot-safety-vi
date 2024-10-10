import pandas as pd
import numpy as np
import re
import torch
from typing import Optional, Union, List
from tqdm import tqdm
from datasets import Dataset
from transformers import pipeline

def check_dataframe(dataframe : pd.DataFrame, 
                    subsets : List[str] = None, 
                    duplicate_subset : List[str] = None,
                    missing_subset : List[str] = None,
                    ) -> None:
    """Check the given DataFrame for duplicate and missing values.

    This function checks for duplicate entries and missing values in the specified subsets of the DataFrame.
    It can ignore certain columns when performing these checks based on the provided parameters.

    Args:
        dataframe (pd.DataFrame): The DataFrame to check for duplicates and missing values.
        subsets (List[str], optional): List of column names to check. If None, checks all columns. Defaults to None.
        duplicate_ignore_subsets (List[str], optional): List of column names to ignore when checking for duplicates. Defaults to None.
        missing_ignore_subsets (List[str], optional): List of column names to ignore when checking for missing values. Defaults to None.

    Print:
        If duplicates or missing values are found in the specified subsets, an assertion error is raised with details.
    """
    if subsets: 
        dataframe = dataframe[subsets]
    _check_duplicate_values(dataframe, duplicate_subset)
    _check_missing_values(dataframe , missing_subset)

def translate(text, tokenizer, model):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  text_token = tokenizer(text, padding=True, max_length=64, truncation=True,return_tensors="pt").to(device)
  text_translate = model.generate(**text_token,num_beams=1, max_length=64, repetition_penalty=2.0 , early_stopping=True)
  text_translate = tokenizer.batch_decode(text_translate, skip_special_tokens=True)
  return text_translate

def lang_detect(df: pd.DataFrame , batch_size: int = 512, thress_hold: float = 0.5, ):
    # Load Pretrained Model
    model_ckpt = "papluca/xlm-roberta-base-language-detection"
    pipe = pipeline("text-classification", model=model_ckpt, device_map="auto")
    ds = Dataset.from_pandas(df = df) # Convert to dataset for maximize efficient for GPU
    keep_langs = ["vi","en"] # Keep English and Vietnamese
    remove_index = [] # for removing
    ambiguous_index = []
    ambiguous_lang = []
    ambiguous_score = []
    for i in tqdm(range(0, len(ds["input"]), batch_size), desc=f"Detecting....",unit="batch"):
        end_index = min(i + batch_size, len(ds["input"]))
        batch = ds[i:end_index]["input"]
        results = pipe(batch, top_k=1, truncation=True)
        for j, res in enumerate(results):
            res = res[0]
            if res["label"] not in keep_langs:
                if res["score"] >= thress_hold:
                    remove_index.append(i + j)
                else:
                    ambiguous_index.append(i+j)
                    ambiguous_lang.append(res["label"])
                    ambiguous_score.append(res["score"])
    # save for consider and remove ambiguous
    ambiguous_data = pd.DataFrame({
        "context" : df.loc[ambiguous_index]["input"],
        "lang" : ambiguous_lang,
        "score" : ambiguous_score
    })
    ambiguous_data.to_csv("ambigous_lang_data.csv")
    print(f"Ambiguous languages data have saved, you can review it now")
    # remove wrong and ambiguous data:
    df.drop(index=remove_index, inplace=True)
    df.drop(index=ambiguous_index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    print(f"Have remove {len(remove_index)} wrong languages and {len(ambiguous_index)} ambiguous languages.")
    return df

def _check_duplicate_values(dataframe : pd.DataFrame , duplicate_subset : List[str] ) -> None:
    if duplicate_subset:
        dataframe = dataframe[duplicate_subset]
    duplicate_count = dataframe.duplicated().sum()
    print(f"There {duplicate_count} duplicate values in the subsets")

def _check_missing_values(dataframe : pd.DataFrame, missing_subset : List[str]) -> None:
    if missing_subset:
        dataframe = dataframe[missing_subset]
    missing_count = dataframe.isna().any().sum()
    print(f"There {missing_count} missing values in the subsets")

def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
    
if __name__== "__main__":
    ...