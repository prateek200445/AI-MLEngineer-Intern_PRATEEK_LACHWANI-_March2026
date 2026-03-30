from datasets import load_dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, context_recall

dataset = load_dataset("json", data_files="ragas_eval_data.json")

score = evaluate(
    dataset=dataset["train"],
    metrics=[
        faithfulness,
        context_recall,
        answer_correctness
    ]
)

df = score.to_pandas()
df.to_csv("ragas_evaluation_results.csv", index=False)
