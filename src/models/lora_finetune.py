import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from src.utils.logging import setup_logging

logger = setup_logging()

def apply_lora(model, rank=8, alpha=16, dropout=0.05):
    try:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout
        )
        return get_peft_model(model, peft_config)
    except Exception as e:
        logger.error(f"Failed to apply LoRA: {e}")
        raise

def train(model, tokenizer, dataset, config):
    model = apply_lora(model, **config["lora_params"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    model.train()
    for epoch in range(config["epochs"]):
        for batch in loader:
            try:
                outputs = model(**batch, labels=batch["labels"])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                logger.error(f"Training step failed: {e}")
        logger.info(f"Epoch {epoch} loss: {loss.item()}")
    return model
