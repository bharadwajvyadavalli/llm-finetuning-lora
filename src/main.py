import argparse
import yaml
import logging
import torch
from src.utils.logging import setup_logging
from src.data.preprocess import load_raw_examples, clean_and_format, tokenize_dataset
from src.models.baseline import load_baseline, test_inference
from src.models.lora_finetune import train
from src.evaluation.evaluate import run_all
from src.utils.trainer import checkpoint

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['preprocess','baseline','train','eval'], required=True)
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    try:
        if args.mode == 'preprocess':
            raw = load_raw_examples(config['data']['input_path'])
            cleaned = [clean_and_format(e) for e in raw]
            tokenizer, _ = load_baseline(config['model']['name'])
            tokenized = tokenize_dataset(cleaned, tokenizer, max_length=config['data']['max_length'])
            torch.save(tokenized, config['data']['output_path'])
            logger.info("Preprocessing complete")
        elif args.mode == 'baseline':
            tok, mdl = load_baseline(config['model']['name'])
            test_inference(tok, mdl, config['test_prompts'])
        elif args.mode == 'train':
            tok, mdl = load_baseline(config['model']['name'])
            dataset = torch.load(config['training']['dataset_path'])
            model = train(mdl, tok, dataset, config['training'])
            checkpoint(model, None, None, config['training']['checkpoint_path'])
        elif args.mode == 'eval':
            tok, mdl = load_baseline(config['model']['name'])
            eval_datasets = {task: torch.load(path) for task, path in config['evaluation']['datasets'].items()}
            results = run_all(mdl, tok, eval_datasets)
            logger.info(f"Evaluation results: {results}")
        else:
            logger.error("Unknown mode")
    except Exception as e:
        logger.error(f"Error during '{args.mode}': {e}")

if __name__ == "__main__":
    main()
