from datasets import load_dataset


wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
wikitext.to_json(f"../source_datasets/wikitext_test.jsonl")
