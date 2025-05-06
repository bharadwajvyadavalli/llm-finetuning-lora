from transformers import LlamaForCausalLM, LlamaTokenizer
from src.utils.logging import setup_logging

logger = setup_logging()

def load_baseline(model_name="meta-llama/Llama-2-7b"):
    try:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        logger.error(f"Failed to load baseline model: {e}")
        raise

def test_inference(tokenizer, model, prompts):
    for p in prompts:
        try:
            inputs = tokenizer(p, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            print(tokenizer.decode(out[0], skip_special_tokens=True))
        except Exception as e:
            logger.error(f"Inference failed for prompt '{p}': {e}")
