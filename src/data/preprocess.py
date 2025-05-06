import json
import torch
from transformers import LlamaTokenizer
from src.utils.logging import setup_logging

logger = setup_logging()

def load_raw_examples(path):
    try:
        with open(path) as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Failed to load raw examples: {e}")
        raise

def clean_and_format(example):
    instruction = example.get('instruction', '').strip()
    input_text = example.get('input', '').strip()
    output = example.get('output', '').strip()
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
    return {"prompt": prompt, "response": output}

def tokenize_dataset(examples, tokenizer: LlamaTokenizer, max_length=1024):
    try:
        return tokenizer(
            [ex["prompt"] for ex in examples],
            [ex["response"] for ex in examples],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise

if __name__ == "__main__":
    raw = load_raw_examples("data/alpaca.json")
    cleaned = [clean_and_format(e) for e in raw]
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
    tokenized = tokenize_dataset(cleaned, tokenizer, max_length=1024)
    torch.save(tokenized, "cache/train.pt")
    logger.info("Preprocessing complete")
