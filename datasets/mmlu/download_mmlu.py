from datasets import load_dataset

dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")
dataset.to_json("mmlu_train.json", lines=True)
