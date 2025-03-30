from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def initialize_model(model_name, device = "cpu"):
    """
    Initialize pretrained model and tokenizer
    
    Args:
        model_name (str): Name of pretrained model from HuggingFace Hub
        device (str): Device placement strategy
        
    Returns:
        tuple: (tokenizer, model) initialized instances
    """
        
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=6, 
        problem_type="multi_label_classification",
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, model