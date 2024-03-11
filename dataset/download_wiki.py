from datasets import load_dataset

train_data = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
train_data.to_json("wikitext103_train.json", lines=True)
