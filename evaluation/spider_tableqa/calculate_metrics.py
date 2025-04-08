import re
import pandas as pd
try:
    from qatch.evaluate_dataset.metrics_evaluators import CellPrecision, CellRecall, ExecutionAccuracy, TupleCardinality, TupleConstraint, TupleOrder
except ImportError as e:
    print(
        f"Module qatch not found. \
          If test Spider, please install it using 'pip install qatch'"
    )


name2evaluator = {
    "cell_precision": CellPrecision(),
    "cell_recall": CellRecall(),
    "tuple_cardinality": TupleCardinality(),
    "tuple_order": TupleOrder(),
    "tuple_constraint": TupleConstraint(),
    "execution_accuracy": ExecutionAccuracy(),
}

def calculate_tableqa_metrics(preds, refs):
    final_metrics = {
        "cell_precision": [],
        "cell_recall": [],
        "tuple_cardinality": [],
        "tuple_order": [],
        "tuple_constraint": [],
        "execution_accuracy": [],
    }
    
    for pred, ref in zip(preds, refs):
        try:
            pred = eval(pred)
            if not isinstance(pred, list):
                print(f"Prediction must be a list of lists: {pred}. Error: {e}")
                for metric in final_metrics:
                    final_metrics[metric].append(0.0)
            if len(pred) == 0:
                pred = [[]]
            elif len(pred) == 1 and not isinstance(pred[0], list):
                pred = [pred]
        except Exception as e:
            print(f"Prediction must be a list of lists: {pred}. Error: {e}")
            for metric in final_metrics:
                final_metrics[metric].append(0.0)
            continue
        
        try:
            ref = eval(ref)
        except Exception as e:
            print(f"Reference must be a list of lists: {ref}. Error: {e}")
            for metric in final_metrics:
                final_metrics[metric].append(0.0)
            continue
        
        for metric in name2evaluator:
            result = name2evaluator[metric].run_metric(prediction=pred, target=ref)
            final_metrics[metric].append(result)
    avg_metrics = {}
    for metric, scores in final_metrics.items():
        if scores:
            avg = sum(scores) / len(scores)
            avg_metrics[metric] = float(avg)
        else:
            avg_metrics[metric] = 0.0
    global_average = float(sum(avg_metrics.values()) / len(avg_metrics)) if avg_metrics else 0.0
    avg_metrics["average_score"] = global_average
    
    return avg_metrics

def calculate_metrics(df: pd.DataFrame) -> dict:
    # This pattern captures text that starts with '[[' and ends with ']]'
    pattern = re.compile(r"(\[\[.*\]\])")
    df["predicted_answer"] = df["predicted_answer"].apply(
        lambda x: pattern.search(x).group(1) if pattern.search(x) else x.strip()
    )
    preds = df["predicted_answer"].tolist()
    refs = df["answer"].tolist()
    score = calculate_tableqa_metrics(preds, refs)
    return score




