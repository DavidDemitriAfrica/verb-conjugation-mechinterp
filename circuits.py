from itertools import product
import random
import math
import torch
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import generate_dataset_per_permutation
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import get_logit_diff

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()

def get_logits(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.cpu()

def compute_logit_diff(logits, correct_word: str, incorrect_word: str):
    correct_id = tokenizer.encode(correct_word, add_prefix_space=True)[0]
    incorrect_id = tokenizer.encode(incorrect_word, add_prefix_space=True)[0]
    logit_correct = logits[0, -1, correct_id].item()
    logit_incorrect = logits[0, -1, incorrect_id].item()
    return logit_correct - logit_incorrect

def path_patch_head(layer_idx: int, head_idx: int, prompt_orig: str, prompt_patch: str):
    # Capture activation from patch prompt.
    inputs_patch = tokenizer(prompt_patch, return_tensors="pt")
    inputs_patch = {k: v.to(device) for k, v in inputs_patch.items()}
    patch_activation = None

    def capture_hook(module, input, output):
        nonlocal patch_activation
        if isinstance(output, tuple):
            output = output[0]
        head_dim = output.shape[-1] // module.num_heads
        act = output[..., head_idx * head_dim:(head_idx + 1) * head_dim].clone()
        if act.dim() == 2:
            act = act.unsqueeze(0)
        patch_activation = act

    handle = model.transformer.h[layer_idx].attn.register_forward_hook(capture_hook)
    _ = model(**inputs_patch)
    handle.remove()
    if patch_activation is None:
        raise RuntimeError("Failed to capture patch activation.")

    # Replace activation on the original prompt.
    def replace_hook(module, input, output):
        tuple_out = False
        if isinstance(output, tuple):
            output_tensor, rest = output[0], output[1:]
            tuple_out = True
        else:
            output_tensor = output
        head_dim = output_tensor.shape[-1] // module.num_heads
        patched = output_tensor.clone()
        batch_size, seq_len, _ = patched.shape
        patch_act = patch_activation if patch_activation.dim() == 3 else patch_activation.unsqueeze(0)
        min_seq_len = min(seq_len, patch_act.shape[1])
        patched[:, :min_seq_len, head_idx * head_dim:(head_idx + 1) * head_dim] = patch_act[:, :min_seq_len, :]
        if tuple_out:
            return (patched,) + rest
        return patched

    handle = model.transformer.h[layer_idx].attn.register_forward_hook(replace_hook)
    inputs_orig = tokenizer(prompt_orig, return_tensors="pt")
    inputs_orig = {k: v.to(device) for k, v in inputs_orig.items()}
    with torch.no_grad():
        outputs_orig = model(**inputs_orig)
    handle.remove()
    return outputs_orig.logits.cpu()

def get_logit_diff_path_patch(layer_idx: int, head_idx: int, prompt: str, prompt_patch: str,
                            correct_word: str, incorrect_word: str):
    patched_logits = path_patch_head(layer_idx, head_idx, prompt, prompt_patch)
    return compute_logit_diff(patched_logits, correct_word, incorrect_word)

singular_names = [
    "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Kara", "Leo", "Mia", "Nina", "Oscar", "Paul", "Quinn", "Rose", "Sam", "Tina"
]
plural_names = [f"{n1} and {n2}" for n1 in singular_names for n2 in singular_names if n1 != n2]
singular_pronouns = ["She", "He"]
plural_pronouns = ["They"]

def generate_counterfactual(example: dict) -> dict:
    new_is_plural = not example["is_plural"]
    if example["is_pronoun"]:
        new_subject = random.choice(plural_pronouns) if new_is_plural else random.choice(singular_pronouns)
    else:
        new_subject = random.choice(plural_names) if new_is_plural else random.choice(singular_names)
    aux = ""
    if example["is_negated"]:
        aux = "did not " if example["tense"] == "past" else ("do not " if new_is_plural else "does not ")
    new_prompt = f"{example['prefix']} {new_subject} {aux}".strip()
    new_example = example.copy()
    new_example["prompt"] = new_prompt
    new_example["is_plural"] = new_is_plural
    new_example["subject"] = new_subject
    new_example["correct_verb"], new_example["incorrect_verb"] = example["incorrect_verb"], example["correct_verb"]
    return new_example

# (a) Zero Ablation.
def attn_zero_hook_factory(head_idx: int):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output_tensor, rest = output[0], output[1:]
            head_dim = output_tensor.shape[-1] // module.num_heads
            patched = output_tensor.clone()
            patched[..., head_idx * head_dim:(head_idx + 1) * head_dim] = 0
            return (patched,) + rest
        else:
            head_dim = output.shape[-1] // module.num_heads
            patched = output.clone()
            patched[..., head_idx * head_dim:(head_idx + 1) * head_dim] = 0
            return patched
    return hook

# (b) Mean Ablation.
def compute_attn_head_mean_activation(layer_idx: int, head_idx: int, dataset, num_samples=50):
    activations = []
    sample_examples = random.sample(dataset, num_samples)
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        head_dim = output.shape[-1] // module.num_heads
        act = output[..., head_idx * head_dim:(head_idx+1)*head_dim].detach().cpu()
        activations.append(act)
    handle = model.transformer.h[layer_idx].attn.register_forward_hook(capture_hook)
    for ex in sample_examples:
        prompt = ex["prompt"]
        _ = get_logits(prompt)
    handle.remove()
    cat_act = torch.cat([a.view(-1, a.shape[-1]) for a in activations], dim=0)
    mean_act = cat_act.mean(dim=0)
    return mean_act

def attn_mean_hook_factory(layer_idx: int, head_idx: int, mean_activation: torch.Tensor):
    def hook(module, input, output):
        tuple_out = False
        if isinstance(output, tuple):
            output_tensor, rest = output[0], output[1:]
            tuple_out = True
        else:
            output_tensor = output
        head_dim = output_tensor.shape[-1] // module.num_heads
        patched = output_tensor.clone()
        batch, seq_len, _ = patched.shape
        mean_expanded = mean_activation.view(1, 1, -1).expand(batch, seq_len, head_dim)
        patched[..., head_idx*head_dim:(head_idx+1)*head_dim] = mean_expanded
        if tuple_out:
            return (patched,) + rest
        return patched
    return hook

# (c) Resampling Ablation.
def compute_attn_head_samples(layer_idx: int, head_idx: int, dataset, num_samples=50, tokens_per_example=5):
    samples = []
    sample_examples = random.sample(dataset, num_samples)
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        head_dim = output.shape[-1] // module.num_heads
        act = output[..., head_idx*head_dim:(head_idx+1)*head_dim].detach().cpu()
        batch, seq_len, _ = act.shape
        indices = random.sample(range(seq_len), min(tokens_per_example, seq_len))
        for i in indices:
            samples.append(act[0, i])
    handle = model.transformer.h[layer_idx].attn.register_forward_hook(capture_hook)
    for ex in sample_examples:
        prompt = ex["prompt"]
        _ = get_logits(prompt)
    handle.remove()
    return samples

def attn_resample_hook_factory(layer_idx: int, head_idx: int, sample_list: list):
    def hook(module, input, output):
        tuple_out = False
        if isinstance(output, tuple):
            output_tensor, rest = output[0], output[1:]
            tuple_out = True
        else:
            output_tensor = output
        head_dim = output_tensor.shape[-1] // module.num_heads
        patched = output_tensor.clone()
        batch, seq_len, _ = patched.shape
        sample = random.choice(sample_list)
        sample_expanded = sample.view(1, 1, -1).expand(batch, seq_len, head_dim)
        patched[..., head_idx*head_dim:(head_idx+1)*head_dim] = sample_expanded
        if tuple_out:
            return (patched,) + rest
        return patched
    return hook

def knockout_attn_heads_method(heads: list, prompt: str, method="zero", attn_means=None, attn_samples=None):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    hook_handles = []
    for (layer, head) in heads:
        if method == "zero":
            handle = model.transformer.h[layer].attn.register_forward_hook(attn_zero_hook_factory(head))
        elif method == "mean":
            if attn_means is None or layer not in attn_means or head not in attn_means[layer]:
                raise ValueError(f"Mean activation not provided for layer {layer}, head {head}")
            handle = model.transformer.h[layer].attn.register_forward_hook(
                attn_mean_hook_factory(layer, head, attn_means[layer][head])
            )
        # Use either "resample" or "interchange" for the resampling method.
        elif method == "resample" or method == "interchange":
            if attn_samples is None or layer not in attn_samples or head not in attn_samples[layer]:
                raise ValueError(f"Resample samples not provided for layer {layer}, head {head}")
            handle = model.transformer.h[layer].attn.register_forward_hook(
                attn_resample_hook_factory(layer, head, attn_samples[layer][head])
            )
        else:
            raise ValueError("Unknown ablation method: " + method)
        hook_handles.append(handle)
    with torch.no_grad():
        outputs = model(**inputs)
    for handle in hook_handles:
        handle.remove()
    return outputs.logits.cpu()

def evaluate_attn_circuit(circuit_heads: list, dataset: list, num_samples: int = 50, method="zero",
                        attn_means=None, attn_samples=None):
    sample_examples = random.sample(dataset, num_samples)
    full_diffs = []
    circuit_diffs = []
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    
    for ex in tqdm(sample_examples, desc="Evaluating circuit examples"):
        baseline_logits = get_logits(ex["prompt"])
        full_diff = compute_logit_diff(baseline_logits, ex["correct_verb"], ex["incorrect_verb"])
        full_diffs.append(full_diff)
        heads_to_knockout = [h for h in all_heads if h not in circuit_heads]
        circuit_logits = knockout_attn_heads_method(heads_to_knockout, ex["prompt"],
                                                    method=method, attn_means=attn_means, attn_samples=attn_samples)
        circuit_diff = compute_logit_diff(circuit_logits, ex["correct_verb"], ex["incorrect_verb"])
        circuit_diffs.append(circuit_diff)
    F_full = sum(full_diffs) / len(full_diffs)
    F_circuit = sum(circuit_diffs) / len(circuit_diffs)
    faithfulness = abs(F_full - F_circuit) / F_full if F_full != 0 else None
    return {"F_full": F_full, "F_circuit": F_circuit, "faithfulness": faithfulness}

def save_results(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_results(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def compute_all_attn_means(dataset, num_samples=50):
    num_layers = len(model.transformer.h)
    num_heads = model.config.n_head
    attn_means = {layer: {} for layer in range(num_layers)}
    for layer in tqdm(range(num_layers), desc="Computing attn means per layer"):
        for head in tqdm(range(num_heads), desc=f"Layer {layer} heads", leave=False):
            mean_act = compute_attn_head_mean_activation(layer, head, dataset, num_samples)
            attn_means[layer][head] = mean_act
            print(f"Computed mean for layer {layer}, head {head}")
    return attn_means

def compute_all_attn_samples(dataset, num_samples=50, tokens_per_example=5):
    num_layers = len(model.transformer.h)
    num_heads = model.config.n_head
    attn_samples = {layer: {} for layer in range(num_layers)}
    for layer in tqdm(range(num_layers), desc="Collecting attn samples per layer"):
        for head in tqdm(range(num_heads), desc=f"Layer {layer} heads", leave=False):
            samples = compute_attn_head_samples(layer, head, dataset, num_samples, tokens_per_example)
            attn_samples[layer][head] = samples
            print(f"Collected {len(samples)} samples for layer {layer}, head {head}")
    return attn_samples

def evaluate_full_model(dataset):
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

def find_minimal_circuit_greedy_by_accuracy(sorted_heads, dataset, threshold=0.05, method="resample"):
    # Evaluate full model accuracy once using evaluate_full_model.
    full_metrics = evaluate_full_model(dataset)
    full_accuracy = full_metrics[0]
    print(f"Full Model Accuracy: {full_accuracy:.4f}")

    circuit_heads = []
    current_accuracy = 0.0

    remaining_heads = sorted_heads.copy()

    while remaining_heads:
        best_head = None
        best_accuracy = current_accuracy
        best_eval = None

        for head in tqdm(remaining_heads, desc="Evaluating candidate heads"):
            candidate_circuit = circuit_heads + [head]
            # Use evaluate_attn_circuit which returns a dictionary
            circuit_metrics = evaluate_attn_circuit(candidate_circuit, dataset, num_samples=50,
                                                    method=method, attn_means=attn_means, attn_samples=attn_samples)
            candidate_accuracy = circuit_metrics["F_circuit"]
            
            if candidate_accuracy > best_accuracy:
                best_accuracy = candidate_accuracy
                best_head = head
                best_eval = circuit_metrics

        if best_head is None:
            print("No head improves accuracy further. Stopping.")
            break

        circuit_heads.append(best_head)
        remaining_heads.remove(best_head)
        current_accuracy = best_accuracy

        print(f"Added head {best_head}. Current Accuracy: {current_accuracy:.4f}")

        # Check if we're within the threshold accuracy of the full model
        accuracy_diff = abs(full_accuracy - current_accuracy)
        if accuracy_diff <= threshold:
            print(f"Desired accuracy threshold reached. Difference: {accuracy_diff:.4f}")
            break

    final_eval = evaluate_attn_circuit(circuit_heads, dataset, num_samples=50, method=method,
                                       attn_means=attn_means, attn_samples=attn_samples)
    return circuit_heads, final_eval

def random_subset(circuit):
    """Return a random subset of the given circuit (each node with probability 0.5)."""
    return [node for node in circuit if random.random() < 0.5]

def evaluate_circuit_metrics(circuit, dataset, num_samples, method, attn_means, attn_samples, num_K_samples=10):
    """
    Evaluates a candidate circuit using three criteria:
    - Faithfulness: How close the circuit's performance is to the full model.
    - Completeness: Maximum incompleteness score over random subsets K ⊆ circuit.
    - Minimality: For each node in the circuit, the maximum drop in performance when that node is removed.
    Returns a dictionary with these metrics.
    """
    full_eval = evaluate_attn_circuit([], dataset, num_samples=num_samples, method=method,
                                    attn_means=attn_means, attn_samples=attn_samples)
    circuit_eval = evaluate_attn_circuit(circuit, dataset, num_samples=num_samples, method=method,
                                        attn_means=attn_means, attn_samples=attn_samples)
    F_full = full_eval["F_full"]
    F_circuit = circuit_eval["F_circuit"]
    faithfulness = abs(F_full - F_circuit) / F_full if F_full != 0 else None

    # Completeness: sample random subsets K and compute the difference.
    completeness_scores = []
    for _ in range(num_K_samples):
        K = random_subset(circuit)
        circuit_minus_K = [node for node in circuit if node not in K]
        eval_circuit_minus_K = evaluate_attn_circuit(circuit_minus_K, dataset, num_samples=num_samples,
                                                    method=method, attn_means=attn_means, attn_samples=attn_samples)
        eval_full_minus_K = evaluate_attn_circuit(K, dataset, num_samples=num_samples,
                                                method=method, attn_means=attn_means, attn_samples=attn_samples)
        incompleteness = abs(eval_circuit_minus_K["F_circuit"] - eval_full_minus_K["F_circuit"])
        completeness_scores.append(incompleteness)
    completeness_metric = max(completeness_scores) if completeness_scores else None

    # Minimality: for each node v, sample random subsets from the remaining nodes.
    minimality_scores = {}
    for v in circuit:
        best_score = 0
        remaining = [node for node in circuit if node != v]
        for _ in range(num_K_samples):
            K = random_subset(remaining)
            eval_with_v = evaluate_attn_circuit([node for node in circuit if node not in K],
                                                dataset, num_samples=num_samples, method=method,
                                                attn_means=attn_means, attn_samples=attn_samples)
            eval_without_v = evaluate_attn_circuit([node for node in circuit if node not in (K + [v])],
                                                dataset, num_samples=num_samples, method=method,
                                                attn_means=attn_means, attn_samples=attn_samples)
            score = abs(eval_with_v["F_circuit"] - eval_without_v["F_circuit"])
            best_score = max(best_score, score)
        minimality_scores[v] = best_score
    avg_minimality = sum(minimality_scores.values()) / len(minimality_scores) if minimality_scores else None
    
    return {
        "faithfulness": faithfulness,
        "completeness": completeness_metric,
        "minimality": minimality_scores,
        "avg_minimality": avg_minimality
    }

def greedy_minimality(circuit, dataset, num_samples, method, attn_means, attn_samples, steps):
    """
    Greedy procedure to build a set K ⊆ circuit that maximizes the incompleteness score.
    At each step, sample a few candidates and add the node that most increases the score.
    """
    K = []
    remaining = circuit.copy()
    k = 5  # Number of candidates to sample at each step.
    for _ in range(steps):
        if not remaining:
            break
        candidate_scores = {}
        candidates = random.sample(remaining, min(k, len(remaining)))
        for v in candidates:
            eval_with_v = evaluate_attn_circuit([node for node in circuit if node not in K],
                                                dataset, num_samples=num_samples, method=method,
                                                attn_means=attn_means, attn_samples=attn_samples)
            eval_without_v = evaluate_attn_circuit([node for node in circuit if node not in (K + [v])],
                                                dataset, num_samples=num_samples, method=method,
                                                attn_means=attn_means, attn_samples=attn_samples)
            candidate_scores[v] = abs(eval_with_v["F_circuit"] - eval_without_v["F_circuit"])
        v_max = max(candidate_scores, key=candidate_scores.get)
        K.append(v_max)
        remaining.remove(v_max)
    return K

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    dataset = generate_dataset_per_permutation(samples_per_perm=10)

    HEAD_EFFECT_CACHE = "results/attn_head_effect_results.pkl"
    HEAD_EFFECT_BASE_CACHE = "results/attn_head_effect_results_base.pkl"
    MINIMAL_CIRCUIT_INTERCHANGE_CACHE = "results/minimal_circuit_resample.pkl"
    MINIMAL_CIRCUIT_BASE_INTERCHANGE_CACHE = "results/minimal_circuit_base_resample.pkl"
    ATTN_MEANS_CACHE = "results/attn_means.pkl"
    ATTN_SAMPLES_CACHE = "results/attn_samples.pkl"

    if os.path.exists(ATTN_MEANS_CACHE):
        with open(ATTN_MEANS_CACHE, "rb") as f:
            attn_means = pickle.load(f)
    else:
        attn_means = compute_all_attn_means(dataset, num_samples=50)
        save_results(ATTN_MEANS_CACHE, attn_means)

    if os.path.exists(ATTN_SAMPLES_CACHE):
        with open(ATTN_SAMPLES_CACHE, "rb") as f:
            attn_samples = pickle.load(f)
    else:
        attn_samples = compute_all_attn_samples(dataset, num_samples=50, tokens_per_example=5)
        save_results(ATTN_SAMPLES_CACHE, attn_samples)

    if os.path.exists(HEAD_EFFECT_CACHE):
        with open(HEAD_EFFECT_CACHE, "rb") as f:
            head_effect_results = pickle.load(f)
    else:
        print("Computing head effects via path patching for each attention head...")
        head_effect_results = []
        sample_dataset = random.sample(dataset, min(20, len(dataset)))  # Slightly more robust sampling
        num_layers = model.config.n_layer
        num_heads = model.config.n_head

        for layer in tqdm(range(num_layers), desc="Layer"):
            for head in tqdm(range(num_heads), desc=f"Layer {layer} Heads", leave=False):
                effects = []
                for ex in sample_dataset:
                    prompt_orig = ex["prompt"]
                    ex_cf = generate_counterfactual(ex)
                    prompt_cf = ex_cf["prompt"]

                    try:
                        # Compute original logit difference (without patching)
                        orig_logits = get_logits(prompt_orig)
                        orig_logit_diff = compute_logit_diff(orig_logits, ex["correct_verb"], ex["incorrect_verb"])

                        # Compute patched logit difference: patch current head from the counterfactual
                        patched_logits = path_patch_head(layer, head, prompt_orig, prompt_cf)
                        patched_logit_diff = compute_logit_diff(patched_logits, ex["correct_verb"], ex["incorrect_verb"])

                        # Effect = change in logit difference due to patching
                        effect = orig_logit_diff - patched_logit_diff
                        effects.append(effect)
                    except Exception as e:
                        print(f"Error at layer {layer}, head {head}: {e}")

                avg_effect = sum(effects) / len(effects) if effects else 0.0
                head_effect_results.append({
                    "layer": layer,
                    "head": head,
                    "avg_effect": avg_effect
                })
        save_results(HEAD_EFFECT_CACHE, head_effect_results)

    # Convert head effects to DataFrame and sort by absolute effect.
    df_effects = pd.DataFrame(head_effect_results)
    df_effects["abs_effect"] = df_effects["avg_effect"].abs()
    df_effects_sorted = df_effects.sort_values("abs_effect", ascending=False)
    print("=== Path Patching Analysis Results (per head) ===")
    print(df_effects_sorted.head(20))  # Print top 20 heads

    sorted_candidate_heads = [
        (int(row["layer"]), int(row["head"]))
        for _, row in df_effects_sorted.iterrows()
    ]

    if not df_effects_sorted.empty:
        heatmap_data = df_effects_sorted.pivot(index='layer', columns='head', values='abs_effect')
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title("Attention Head Absolute Effects")
        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.xticks(range(heatmap_data.shape[1]), heatmap_data.columns)
        plt.yticks(range(heatmap_data.shape[0]), heatmap_data.index)
        plt.savefig("attn_heatmap.png")
        plt.close()
        print("Saved heatmap as attn_heatmap.png")

    print("\nSearching minimal circuit using resample ablation ...")
    cached_result = load_results(MINIMAL_CIRCUIT_INTERCHANGE_CACHE)
    if cached_result is not None:
        minimal_circuit, minimal_eval = cached_result
        print("Loaded cached result for resample ablation.")
    else:
        minimal_circuit, minimal_eval = find_minimal_circuit_greedy_by_accuracy(
            sorted_candidate_heads, dataset, threshold=0.01, method="resample"
        )
        save_results(MINIMAL_CIRCUIT_INTERCHANGE_CACHE, (minimal_circuit, minimal_eval))
    print("\nMinimal circuit found using greedy search and resample ablation:")
    print(minimal_circuit)
    print("Circuit evaluation:", minimal_eval)

    # Evaluate robust metrics on the found circuit.
    metrics = evaluate_circuit_metrics(minimal_circuit, dataset, num_samples=50, method="resample",
                                    attn_means=attn_means, attn_samples=attn_samples, num_K_samples=10)
    print("\nRobust evaluation for resample ablation:")
    print("Faithfulness:", metrics["faithfulness"])
    print("Completeness (max incompleteness):", metrics["completeness"])
    print("Average Minimality:", metrics["avg_minimality"])

    greedy_K = greedy_minimality(minimal_circuit, dataset, num_samples=50, method="resample",
                                attn_means=attn_means, attn_samples=attn_samples, steps=len(minimal_circuit))
    print("Greedy minimality set K:", greedy_K)

    print("\n=== Iterative Circuit Enhancement ===")
    # Use the minimal circuit from resample ablation as the base circuit.
    base_circuit = minimal_circuit.copy()
    base_eval = minimal_eval
    print("Base minimal circuit (Interchange Ablation):", base_circuit)
    print("Base evaluation:", base_eval)

    # Iteratively add one head at a time from the remaining candidates.
    iterative_results = []
    remaining_heads = [h for h in sorted_candidate_heads if h not in base_circuit]
    current_circuit = base_circuit.copy()
    current_eval = base_eval
    improvement = True
    iteration = 0
    max_iterations = 100  # adjust as desired
    while improvement and iteration < max_iterations and remaining_heads:
        improvement = False
        best_candidate = None
        best_eval = None
        best_diff = float('inf')
        for head in remaining_heads:
            candidate = current_circuit + [head]
            eval_candidate = evaluate_attn_circuit(candidate, dataset, num_samples=50, method="resample",
                                                attn_means=attn_means, attn_samples=attn_samples)
            diff = abs(eval_candidate["F_full"] - eval_candidate["F_circuit"]) / eval_candidate["F_full"]
            if diff < best_diff:
                best_diff = diff
                best_candidate = candidate
                best_eval = eval_candidate
        # Check if the best candidate improves performance compared to current circuit.
        current_diff = abs(current_eval["F_full"] - current_eval["F_circuit"]) / current_eval["F_full"]
        if best_candidate is not None and best_diff < current_diff:
            current_circuit = best_candidate
            current_eval = best_eval
            # Remove heads that have been added.
            remaining_heads = [h for h in remaining_heads if h not in best_candidate]
            iterative_results.append((current_circuit.copy(), current_eval.copy()))
            improvement = True
            print(f"Iteration {iteration+1}: Added head, new relative diff = {best_diff:.4f}")
        iteration += 1

    print("\nFinal iterative circuit:", current_circuit)
    print("Final evaluation:", current_eval)

    # Save combined results.
    with open("all_minimal_circuits_and_metrics.pkl", "wb") as f:
        pickle.dump({"minimal_results": {"resample": (minimal_circuit, minimal_eval)},
                    "robust_metrics": {"resample": metrics},
                    "iterative_results": iterative_results}, f)
    print("Saved all minimal circuit results and robust metrics as all_minimal_circuits_and_metrics.pkl")