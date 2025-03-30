from transformers import Trainer, DataCollatorForLanguageModeling

def setup_trainer(model, train_dataset, val_dataset, tokenizer, args):
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    return trainer