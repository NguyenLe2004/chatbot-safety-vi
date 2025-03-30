import argparse
from dotenv import load_dotenv
import os
from transformers import TrainingArguments
from huggingface_hub import login

from data.dataset import load_and_preprocess_data
from model.trainer import setup_trainer
from model.model import initialize_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Safety Classification Model")
    parser.add_argument("--data-path", type=str, default="chatbot-vi/ChabotVi-Final-Data")
    parser.add_argument("--model-name", type=str, default="pengold/distilbert-base-vietnamese-case")
    parser.add_argument("--num-labels", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="safety_outputs")
    
    return parser.parse_args()

def main(args):
    load_dotenv()
    login(token = os.getenv("hf_token"))

    # Initialize components
    tokenizer, model = initialize_model(args.model_name)
    
    # Load and preprocess data
    train_data, val_data = load_and_preprocess_data(
        args.data_path,
        tokenizer
    )
    
    # Training setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=True,
        use_mps_device = False,
    )
    # training_args.place_model_on_device = False
    
    trainer = setup_trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data
    )
    
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)