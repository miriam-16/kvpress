import re
import pandas as pd
from qatch.connectors import SqliteConnector
from qatch.evaluate_dataset.orchestrator_evaluator import OrchestratorEvaluator
from qatch.evaluate_dataset.metrics_evaluators import cell_precision, cell_recall, tuple_cardinality, tuple_order, tuple_constraint, execution_accuracy

def test_evaluate_orchestrator(preds,refs,db_names):

    orchestrator = OrchestratorEvaluator(
        evaluator_names=[
            "cell_precision",
            "cell_recall",
            "tuple_cardinality",
             #"tuple_order",
            "tuple_constraint",
            "execution_accuracy",
        ]
    )

    #create a dictionary which stores each metric defined in evaluator names

    final_metrics={"cell_precision":[],
            "cell_recall":[],
            "tuple_cardinality":[],
            #"tuple_order":[],
            "tuple_constraint":[],
            "execution_accuracy":[],}

    for pred, ref,db_name in zip(preds,refs,db_names): 
        try:
            pred = eval(pred)
        except Exception as e:
            print(f"prediction must be a List of lists: {pred}. Error: {e}")
            result = {name: 0.0 for name in orchestrator.evaluator_names}
        else:
            ref= eval(ref)

            result = orchestrator.evaluate_single_test(
            target_query=ref,
            predicted_query=pred,
            connector=SqliteConnector(relative_db_path=rf"C:\Users\elyfa\OneDrive - eurecom.fr\SEMESTER_PROJECT\spider_data\spider_data\test_database\{db_name}\{db_name}.sqlite", db_name=db_name),
        )
        
        #result is a dictionary, store each value in the list defined in final_etrix 
        final_metrics["cell_precision"].append(result["cell_precision"])
        final_metrics["cell_recall"].append(result["cell_recall"])
        final_metrics["tuple_cardinality"].append(result["tuple_cardinality"])
        final_metrics["tuple_constraint"].append(result["tuple_constraint"])
        final_metrics["execution_accuracy"].append(result["execution_accuracy"])
    #calculate the average of each metric
    result = {}
    for key in final_metrics.keys():
        print(key)
        print(final_metrics[key])
        result[key] = sum(final_metrics[key])/len(final_metrics[key])
    
    #compute mean among different metrics
    final_score = sum(result.values())/len(result)
    print("final_score: ",final_score)
    return final_score




def calculate_metrics(df: pd.DataFrame) -> dict:
    print("iniside calculate_metrics")
    np_pattern = re.compile(r"[\x00-\x1f]")
    df["predicted_answer"] = df["predicted_answer"].apply(lambda x: np_pattern.sub("", x.strip()).strip())

    preds = df["predicted_answer"].tolist()
    refs = df["answer"].tolist()
    db_names = df["db_id"].tolist()

    preds=["[['a', 'b'], ['c', '']]","[['a', 'b'], ['c', 'd']]"]
    refs=["[['a', 'b'], ['c', 'd']]","[['a', 'b'], ['c', 'd']]"]
    
    score = test_evaluate_orchestrator(preds, refs, db_names)
    

    return score




