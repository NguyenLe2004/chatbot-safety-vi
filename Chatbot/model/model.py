from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def initialize_model(model_name, device_map="cpu"):
    """
    Initialize pretrained model and tokenizer
    
    Args:
        model_name (str): Name of pretrained model from HuggingFace Hub
        device_map (str): Device placement strategy
        
    Returns:
        tuple: (tokenizer, model) initialized instances
    """
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        device_map=device_map
    )
    return tokenizer, model

def freeze_layers(model, num_frozen_blocks=29):
    """
    Freeze specified transformer blocks
    
    Args:
        model (PreTrainedModel): Model to modify
        num_frozen_blocks (int): Number of blocks to freeze
        
    Returns:
        PreTrainedModel: Model with frozen layers
    """
    for name, param in model.named_parameters():
        if "transformer.blocks." in name:
            block_id = int(name.split("transformer.blocks.")[1].split(".")[0])
            param.requires_grad = block_id >= num_frozen_blocks
        elif "norm_f" in name :
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model have {trainable_params} trainable parameters")
    return model
