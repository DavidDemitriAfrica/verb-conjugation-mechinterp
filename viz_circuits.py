import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from circuits import knockout_attn_heads_method

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cumulative_results_file = "iterative_search_by_setting_results.pkl"
with open(cumulative_results_file, "rb") as f:
    cumulative_data = pickle.load(f)
setting_results = cumulative_data.get("setting_results", {})

with open("results/attn_means.pkl", "rb") as f:
    attn_means = pickle.load(f)
with open("results/attn_samples.pkl", "rb") as f:
    attn_samples = pickle.load(f)

from dataset import generate_dataset_per_permutation
dataset = generate_dataset_per_permutation(samples_per_perm=5, seed=69)

def get_logits(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure inputs on device
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

def compute_accuracy_from_logits(logits, correct_word: str, incorrect_word: str):
    correct_ids = tokenizer.encode(correct_word, add_prefix_space=True)
    incorrect_ids = tokenizer.encode(incorrect_word, add_prefix_space=True)
    logit_correct = sum(logits[0, -1, token_id].item() for token_id in correct_ids)
    logit_incorrect = sum(logits[0, -1, token_id].item() for token_id in incorrect_ids)
    return logit_correct > logit_incorrect

def evaluate_full_model_accuracy(filtered_ds):
    correct = 0
    total = 0
    for ex in tqdm(filtered_ds, desc="Evaluating full model accuracy"):
        logits = get_logits(ex["prompt"])
        if compute_accuracy_from_logits(logits, ex["correct_verb"], ex["incorrect_verb"]):
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def evaluate_circuit_accuracy(circuit, filtered_ds, method, attn_means, attn_samples):
    correct = 0
    total = 0
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    knockout_heads = [h for h in all_heads if h not in circuit]
    
    for ex in tqdm(filtered_ds, desc=f"Evaluating circuit (size {len(circuit)}) accuracy for method '{method}'"):
        logits = knockout_attn_heads_method(
            knockout_heads, ex["prompt"], method=method,
            attn_means=attn_means, attn_samples=attn_samples
        )
        if compute_accuracy_from_logits(logits, ex["correct_verb"], ex["incorrect_verb"]):
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def filter_dataset_by_setting(dataset, setting):
    """
    Filter the dataset based on the given setting.
    Keys assumed in each example:
      - "is_plural" (bool)
      - "is_negated" (bool)
      - "prefix" (string)
      - "is_pronoun" (bool)
      - "tense" (string, e.g., "past" or "present")
      - "use_irregular" (bool)
    For the "base" setting, we require singular, non-negated, and no prefix.
    """
    if setting == "base":
        return [ex for ex in dataset if (ex.get("is_plural", False) is False and 
                                         ex.get("is_negated", False) is False and 
                                         ex.get("prefix", "").strip() == "")]
    elif setting == "plural":
        return [ex for ex in dataset if ex.get("is_plural", False) is True]
    elif setting == "negation":
        return [ex for ex in dataset if ex.get("is_negated", False) is True]
    elif setting == "prefix":
        return [ex for ex in dataset if ex.get("prefix", "").strip() != ""]
    elif setting == "pronoun":
        return [ex for ex in dataset if ex.get("is_pronoun", False) is True]
    elif setting == "past_tense":
        return [ex for ex in dataset if ex.get("tense", "present") == "past"]
    elif setting == "irregular":
        return [ex for ex in dataset if ex.get("use_irregular", False) is True]
    elif setting == "complex":
        return [
            ex for ex in dataset 
            if ex.get("is_plural", False) is True and 
               ex.get("is_negated", False) is True and 
               ex.get("prefix", "").strip() != "" and 
               ex.get("is_pronoun", False) is True and 
               ex.get("tense", "present") == "past" and 
               ex.get("use_irregular", False) is True
        ]
    else:
        return dataset

results_data = []
for setting, res in setting_results.items():
    # Filter dataset for this setting.
    filtered_ds = filter_dataset_by_setting(dataset, setting)
    print(f"\nSetting '{setting}': Filtered dataset size = {len(filtered_ds)}")
    
    full_acc_setting = evaluate_full_model_accuracy(filtered_ds)
    circuit = res["circuit"]
    circuit_acc_setting = evaluate_circuit_accuracy(circuit, filtered_ds, method="resample", 
                                                    attn_means=attn_means, attn_samples=attn_samples)
    # Also record evaluation metrics from the iterative search (if desired)
    eval_metrics = res["eval"]
    results_data.append({
        "Setting": setting,
        "Circuit": str(circuit),
        "Circuit Size": len(circuit),
        "Full Model Accuracy": full_acc_setting,
        "Circuit Accuracy": circuit_acc_setting,
        "F_full": eval_metrics["F_full"],
        "F_circuit": eval_metrics["F_circuit"],
        "Faithfulness": eval_metrics["faithfulness"]
    })

df_results = pd.DataFrame(results_data)
print("\nSummary of Evaluation Metrics per Setting:")
print(df_results)

plt.figure(figsize=(10, 6))
x = range(len(df_results))
bar_width = 0.35

# Full model accuracy (evaluated on the filtered subset for each setting)
full_accs = df_results["Full Model Accuracy"].tolist()
plt.bar(x, full_accs, width=bar_width, label="Full Model Accuracy")

# Circuit accuracy (evaluated on the same filtered subset)
circuit_accs = df_results["Circuit Accuracy"].tolist()
plt.bar([p + bar_width for p in x], circuit_accs, width=bar_width, label="Circuit Accuracy")

plt.xlabel("Setting")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison (Filtered by Setting)")
plt.xticks([p + bar_width / 2 for p in x], df_results["Setting"])
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_comparison_per_setting.png")
plt.show()

# =======================================
# Save summary results as CSV.
# =======================================
df_results.to_csv("circuit_accuracy_results_per_setting.csv", index=False)
print("Saved evaluation summary to 'circuit_accuracy_results_per_setting.csv'")
