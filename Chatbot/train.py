import argparse
from dotenv import load_dotenv
import os
import torch
from huggingface_hub import login
from transformers.trainer import TrainingArguments
from data.dataset import load_and_preprocess_data
from model.model import initialize_model, freeze_layers
from model.trainer import setup_trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Vietnamese Chatbot Model")
    
    # Data arguments
    parser.add_argument("--output-dir", type=str, default="chatbot",
                       help="Path to output directory")
    
    parser.add_argument("--data-path", type=str, default="chatbot-vi/ChabotVi-Final-Data",
                       help="Path to training data file")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data for validation/testing")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="vinai/PhoGPT-4B-Chat" ,
                       help="Pretrained model name from HuggingFace")
    parser.add_argument("--num-frozen-blocks", type=int, default=29,
                       help="Number of transformer blocks to freeze")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--fp16", action="store_true",
                       help="Enable mixed precision training")
    
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv()
    login(token = os.getenv("hf_token"))
    # Setup environment
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize components
    tokenizer, model = initialize_model(args.model_name)
    model = freeze_layers(model, args.num_frozen_blocks)
    
    # Load and preprocess data
    train_dataset, val_dataset = load_and_preprocess_data(
        args.data_path, 
        tokenizer,
        test_size=args.test_size
    )
    
    # Train model
    args = TrainingArguments(
        "chatbot-vi",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_steps=10,
        optim="adamw_torch_fused", 
        lr_scheduler_type="constant",
        save_total_limit=1,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        fp16_full_eval=args.fp16,
        save_only_model=True, 
        push_to_hub=False,
    )
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=args.output_dir,
        args = args
    )
    trainer.train()
    
    # Save results
    torch.save(trainer.model.state_dict(), 
                os.path.join(args.output_dir, "model_weights.pth"))
    print("Training completed successfully")

if __name__ == "__main__":
    main()