import pickle
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from utils import get_logit_diff
from circuits import knockout_attn_heads_method

def generate_simple_dataset(samples=100, seed=42):
    """
    Generate a dataset with a fixed base condition:
      - is_plural: False (singular)
      - is_negated: False (affirmative)
      - has_prefix: False (no prefix)
      - is_pronoun: False (use a name)
      - tense: "present"
      - use_irregular: False (regular verb)
    """
    random.seed(seed)
    dataset = []
    
    # Fixed settings.
    is_plural = False
    is_negated = False
    has_prefix = False
    is_pronoun = False
    tense = "present"
    use_irregular = False
    
    singular_names = ["Alice", "Bob", "Charlie"]
    # For a singular subject in present tense, we use the singular regular verb.
    reg_present_singular = ["walks"]
    # Use the plural form as an incorrect alternative.
    reg_present_plural = ["walk"]
    
    prefix = ""  # No prefix.
    
    for _ in range(samples):
        subject = random.choice(singular_names)
        correct_verb = random.choice(reg_present_singular)
        incorrect_verb = random.choice(reg_present_plural)
        prompt = f"{prefix}{subject} "  # e.g., "Alice "
        
        dataset.append({
            "prompt": prompt,
            "correct_verb": correct_verb,
            "incorrect_verb": incorrect_verb,
            "is_negated": is_negated,
            "is_plural": is_plural,
            "subject": subject,
            "prefix": prefix.strip(),
            "is_pronoun": is_pronoun,
            "tense": tense,
            "use_irregular": use_irregular
        })
    return dataset

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Load attention statistics used in the knockout function.
with open("results/attn_means.pkl", "rb") as f:
    attn_means = pickle.load(f)
with open("results/attn_samples.pkl", "rb") as f:
    attn_samples = pickle.load(f)

def evaluate_full_model(dataset):
    """
    Compute predictions using get_logit_diff and then use sklearn metrics.
    """
    predictions = []
    golds = []
    for ex in tqdm(dataset, desc="Evaluating full model accuracy"):
        # Compute logit difference using the helper.
        logit_diff, _, _ = get_logit_diff(
            ex["prompt"], ex["correct_verb"], ex["incorrect_verb"], tokenizer, model
        )
        pred = 1 if logit_diff > 0 else 0
        predictions.append(pred)
        # Gold label is 1 (the model should prefer the correct verb).
        golds.append(1)
    return (accuracy_score(golds, predictions),
            precision_score(golds, predictions, zero_division=0),
            recall_score(golds, predictions, zero_division=0),
            f1_score(golds, predictions, zero_division=0))

def evaluate_circuit_accuracy(circuit, dataset, method, attn_means, attn_samples):
    """
    Evaluate model performance after knocking out all heads not in 'circuit'.
    Computes predictions similarly to evaluate_full_model.
    """
    predictions = []
    golds = []
    # Identify all heads in the model.
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    # Knockout all heads that are not in the circuit.
    knockout_heads = [h for h in all_heads if h not in circuit]
    
    for ex in tqdm(dataset, desc=f"Evaluating circuit accuracy (method: '{method}')"):
        # Get logits with knockout applied.
        logits = knockout_attn_heads_method(
            knockout_heads, ex["prompt"], method=method,
            attn_means=attn_means, attn_samples=attn_samples
        )
        # Manually compute the logit difference (mirroring get_logit_diff).
        correct_id = tokenizer.encode(ex["correct_verb"], add_prefix_space=True)[0]
        incorrect_id = tokenizer.encode(ex["incorrect_verb"], add_prefix_space=True)[0]
        logit_diff = logits[0, -1, correct_id].item() - logits[0, -1, incorrect_id].item()
        pred = 1 if logit_diff > 0 else 0
        predictions.append(pred)
        golds.append(1)
        
    return (accuracy_score(golds, predictions),
            precision_score(golds, predictions, zero_division=0),
            recall_score(golds, predictions, zero_division=0),
            f1_score(golds, predictions, zero_division=0))

with open("results/minimal_circuit_resample.pkl", "rb") as f:
    base_circuit = pickle.load(f)
# Assume the method used to obtain the circuit was "resample".
method_name = "resample"

# Generate the simple dataset.
dataset = generate_simple_dataset(samples=50, seed=42)

# Evaluate full model using get_logit_diff.
full_metrics = evaluate_full_model(dataset)
print(f"Full Model Metrics (Accuracy, Precision, Recall, F1): {full_metrics}")

# Evaluate circuit accuracy.
circuit_metrics = evaluate_circuit_accuracy(base_circuit, dataset, method_name, attn_means, attn_samples)
print(f"Base Circuit Metrics (Accuracy, Precision, Recall, F1): {circuit_metrics}")


methods = ["Full Model", "Base Circuit"]
accuracies = [full_metrics[0], circuit_metrics[0]]

plt.figure(figsize=(6, 4))
bar_width = 0.4
x = range(len(methods))
plt.bar(x, accuracies, width=bar_width)
plt.xticks(x, methods)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison on Simple Dataset")
plt.ylim(0, 1)
plt.show()
