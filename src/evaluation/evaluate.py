from datasets import load_metric
from src.utils.logging import setup_logging

logger = setup_logging()

def evaluate_qa(preds, refs):
    try:
        acc = load_metric("accuracy")
        return acc.compute(predictions=preds, references=refs)
    except Exception as e:
        logger.error(f"QA evaluation failed: {e}")
        raise

def evaluate_summarization(preds, refs):
    try:
        rouge = load_metric("rouge")
        return rouge.compute(predictions=preds, references=refs)
    except Exception as e:
        logger.error(f"Summarization evaluation failed: {e}")
        raise

def evaluate_classification(preds, refs):
    try:
        acc = load_metric("accuracy")
        return acc.compute(predictions=preds, references=refs)
    except Exception as e:
        logger.error(f"Classification evaluation failed: {e}")
        raise

def evaluate_translation(preds, refs):
    try:
        bleu = load_metric("bleu")
        return bleu.compute(predictions=[p.split() for p in preds], references=[[r.split()] for r in refs])
    except Exception as e:
        logger.error(f"Translation evaluation failed: {e}")
        raise

def run_all(model, tokenizer, eval_datasets):
    results = {}
    for task, ds in eval_datasets.items():
        preds, refs = [], []
        for ex in ds:
            try:
                inp = tokenizer(ex["prompt"], return_tensors="pt")
                out = model.generate(**inp)
                preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
                refs.append(ex["response"])
            except Exception as e:
                logger.error(f"Inference for {task} failed: {e}")
        if task == "qa":
            results[task] = evaluate_qa(preds, refs)
        elif task == "summarization":
            results[task] = evaluate_summarization(preds, refs)
        elif task == "classification":
            results[task] = evaluate_classification(preds, refs)
        elif task == "translation":
            results[task] = evaluate_translation(preds, refs)
        else:
            logger.warning(f"No evaluator for task {task}")
    return results
