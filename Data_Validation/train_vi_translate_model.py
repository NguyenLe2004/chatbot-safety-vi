import torch 
import torch.nn as nn
import pandas as pd
import argparse
import wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import GenerationConfig, EarlyStoppingCallback, T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from config import WANDB_KEY
from warnings import filterwarnings

def get_args():
    parser = argparse.ArgumentParser(description="Train Translate model")
    parser.add_argument("--batch_size","-b", type=int, default=32)
    parser.add_argument("--logging_steps","-ls", type=int, default=35)
    parser.add_argument("--epochs","-e",type=int, default=100)
    parser.add_argument("--learning_rate","-lr",type=float, default=1e-4)
    parser.add_argument("--patience","-p", type=int, default=5)
    args = parser.parse_args()
    return args

def main(args):
    wandb.login(key=WANDB_KEY)
    tokenizer, model, generation_config = load_model("NlpHUST/t5-en-vi-base")
    generation_config.repetition_penalty = 2.0
    generation_config.max_length = 64
    generation_config.num_beams = 1
    train_dataset, val_dataset = load_dataset(data_path="viHOS_Cleaned_For_Translate.csv",tokenizer = tokenizer,val_size=0.2)
    trainer = model_preprocess(model=model,
                               tokenizer=tokenizer,
                               generation_config=generation_config,
                               train_dataset=train_dataset,val_dataset=val_dataset,
                               batch_size=args.batch_size,
                               lr=args.learning_rate,
                               epochs=args.epochs,
                               logging_steps=args.logging_steps,
                               patience=args.patience
                               )
    trainer.train()    


def model_preprocess(model, tokenizer, generation_config, train_dataset, val_dataset,batch_size, lr, epochs, logging_steps, patience  ) -> Seq2SeqTrainer:
    args = Seq2SeqTrainingArguments(
        "Translate_hate_offensive_model",
        eval_strategy = "epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        lr_scheduler_type = "reduce_lr_on_plateau",
        lr_scheduler_kwargs = {"mode":'min', "factor":0.2, "patience":3},  
        save_total_limit = 2,
        load_best_model_at_end=True,
        num_train_epochs = epochs,
        predict_with_generate = True,
        bf16=True,
        logging_steps = logging_steps,
        metric_for_best_model='eval_loss',
        generation_config = generation_config
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset = train_dataset ,
        eval_dataset = val_dataset,
        tokenizer= tokenizer,
        data_collator = data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]
    )
    return trainer

def load_model(model_ckp : str) ->tuple[T5Tokenizer, T5ForConditionalGeneration, GenerationConfig]:
    tokenizer = T5Tokenizer.from_pretrained(model_ckp)  
    model = T5ForConditionalGeneration.from_pretrained(model_ckp, device_map="auto")
    # Freeze layer
    for name, param in model.named_parameters():
        if name.startswith(("encoder.block")):
            param.requires_grad = False
        if name.startswith(("encoder.block.11","encoder.block.10","encoder.block.9")):
            param.requires_grad = True
    print(model.num_parameters(only_trainable=True))
    return tokenizer, model,  model.generation_config

def load_dataset(data_path : str, tokenizer ,  val_size : float) -> tuple[Dataset, Dataset, Dataset]: 
    df = pd.read_csv(data_path)
    train_dataset, val_dataset = train_val_split(df, val_size=val_size)
    train_dataset = train_dataset.map(lambda ds: preprocess_function(ds, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda ds: preprocess_function(ds, tokenizer), batched=True)
    return train_dataset, val_dataset


def train_val_split(dataframe: pd.DataFrame, val_size : float) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    train_df, val_df = train_test_split(dataframe,test_size=val_size)
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    return train_dataset, val_dataset

def preprocess_function(ds: Dataset, tokenizer, max_length : int = 64, truncation : bool = True) ->dict:
    inputs = ds['translate']
    outputs = ds['content']
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=truncation, padding=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, max_length=max_length, truncation=truncation, padding=True)
    return {
        'input_ids': model_inputs['input_ids'],
        'attention_mask': model_inputs['attention_mask'],
        'labels': labels['input_ids']
    }
if __name__ == "__main__" :
    filterwarnings("ignore")
    args = get_args()
    main(args)