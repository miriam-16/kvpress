import pandas as pd
import evaluate
from autoacu import A3CU

def get_sacrebleu_scores(predictions, references):
    # Load the sacreBLEU metric and compute the score.
    sacrebleu = evaluate.load("sacrebleu")
    # Note: The evaluate package expects references as a list of lists.
    results = sacrebleu.compute(predictions=predictions, references=[[r] for r in references])
    return results["score"]

def get_rougel_scores(predictions, references):
    # Load the ROUGE metric and compute the rougeL score.
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results["rougeL"]

def get_meteor_scores(predictions, references):
    # Load the METEOR metric and compute the score.
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predictions, references=references)
    return results["meteor"]

def get_bert_scores(predictions, references):
    # Load BERTScore and compute the average F1 score.
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    # Return the average F1 score.
    return sum(results["f1"]) / len(results["f1"])

def get_autoacu_scores(predictions, references):
    # Initialize AutoACU (with the default device as 0; adjust if needed)
    a3cu = A3CU(device=0)
    recall_scores, prec_scores, f1_scores = a3cu.score(
        references=references,
        candidates=predictions,
        batch_size=32,
        output_path=None,
    )
    # Return the average F1 score from AutoACU.
    return sum(f1_scores) / len(f1_scores)

def calculate_qtsumm_metrics(predictions, references):
    """
    Compute a collection of evaluation metrics for summary generation.
    This function returns a dictionary with the metric name as key and the computed score as value.
    TAPAS score is ignored as per the requirements.
    """
    metrics = {}
    metrics["sacreBLEU"] = get_sacrebleu_scores(predictions, references)
    metrics["Rouge-L"] = get_rougel_scores(predictions, references)
    metrics["METEOR"] = get_meteor_scores(predictions, references)
    metrics["BERTScore"] = get_bert_scores(predictions, references)
    metrics["AutoACU"] = get_autoacu_scores(predictions, references)
    return metrics

def calculate_metrics(df: pd.DataFrame) -> dict:
    preds = df["predicted_answer"].tolist()
    refs = df["answer"].tolist()
    score = calculate_qtsumm_metrics(preds, refs)
    return score