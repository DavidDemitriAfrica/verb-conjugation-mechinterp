import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product
from dataset import generate_dataset_per_permutation
from utils import get_logit_diff

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Generate dataset with 100 samples per permutation.
dataset = generate_dataset_per_permutation(samples_per_perm=100)

# Compute model predictions.
for example in dataset:
    logit_diff, logit_correct, logit_incorrect = get_logit_diff(
        example["prompt"], example["correct_verb"], example["incorrect_verb"], tokenizer, model
    )
    example["logit_diff"] = logit_diff
    example["logit_correct"] = logit_correct
    example["logit_incorrect"] = logit_incorrect
    example["prediction"] = 1 if logit_diff > 0 else 0
    example["gold"] = 1

# Convert dataset to DataFrame.
df = pd.DataFrame(dataset)

# Add a flag for prefix presence (has_prefix is already fixed but we add it for grouping).
df["has_prefix"] = df["prefix"].apply(lambda x: bool(x))

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Print metrics for each permutation.
group_cols = ["is_plural", "is_negated", "has_prefix", "is_pronoun", "tense", "use_irregular"]

print("=== Metrics for Each Permutation of Conditions (100 samples each) ===\n")
grouped = df.groupby(group_cols)
for name, group in grouped:
    conditions = {
        "is_plural": "Plural" if name[0] else "Singular",
        "is_negated": "Negated" if name[1] else "Affirmative",
        "has_prefix": "With Prefix" if name[2] else "Without Prefix",
        "is_pronoun": "Pronoun" if name[3] else "Name",
        "tense": name[4],
        "use_irregular": "Irregular" if name[5] else "Regular"
    }
    metrics = compute_metrics(group["gold"], group["prediction"])
    print("Conditions:", conditions)
    print("Sample Count:", len(group))
    print("Metrics:", metrics)
    print("-" * 60)
