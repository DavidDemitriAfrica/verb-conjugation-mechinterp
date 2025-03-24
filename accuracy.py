import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import generate_dataset_per_permutation
from utils import get_logit_diff
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Generate dataset with 100 samples per permutation.
dataset = generate_dataset_per_permutation(samples_per_perm=100, seed=69)

# Compute model predictions.
for example in dataset:
    logit_diff, _, _ = get_logit_diff(
        example["prompt"], example["correct_verb"], example["incorrect_verb"], tokenizer, model
    )
    example["prediction"] = 1 if logit_diff > 0 else 0
    example["gold"] = 1

# Convert dataset to DataFrame.
df = pd.DataFrame(dataset)

df["has_prefix"] = df["prefix"].apply(lambda x: bool(x))
group_cols = ["is_plural", "is_negated", "has_prefix", "is_pronoun", "tense", "use_irregular"]

# Compute metrics and prepare LaTeX table
rows = []
grouped = df.groupby(group_cols)
for name, group in grouped:
    conditions = {
        "Number": "Plural" if name[0] else "Singular",
        "Negation": "Negated" if name[1] else "Affirmative",
        "Prefix": "With" if name[2] else "Without",
        "Subject": "Pronoun" if name[3] else "Name",
        "Tense": name[4].capitalize(),
        "Verb Type": "Irregular" if name[5] else "Regular"
    }
    metrics = {
        "Accuracy": accuracy_score(group["gold"], group["prediction"]),
        "Precision": precision_score(group["gold"], group["prediction"], zero_division=0),
        "Recall": recall_score(group["gold"], group["prediction"], zero_division=0),
        "F1 Score": f1_score(group["gold"], group["prediction"], zero_division=0)
    }
    row = {**conditions, **metrics}
    rows.append(row)

results_df = pd.DataFrame(rows)
latex_table = results_df.to_latex(index=False, float_format="%.2f")

print(latex_table)
