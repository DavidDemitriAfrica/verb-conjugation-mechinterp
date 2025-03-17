import random
import torch
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import generate_dataset_per_permutation

# =======================================
# Load GPT-2 small and set to eval mode.
# =======================================
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# =======================================
# Generate dataset (100 samples per permutation)
# =======================================
dataset = generate_dataset_per_permutation(samples_per_perm=100)

# =======================================
# Define cache file paths.
# =======================================
HEAD_EFFECT_CACHE = "results/attn_head_effect_results.pkl"
MINIMAL_CIRCUIT_ZERO_CACHE = "results/minimal_circuit_zero.pkl"
MINIMAL_CIRCUIT_MEAN_CACHE = "results/minimal_circuit_mean.pkl"
MINIMAL_CIRCUIT_RESAMPLE_CACHE = "results/minimal_circuit_resample.pkl"
ATTN_MEANS_CACHE = "results/attn_means.pkl"
ATTN_SAMPLES_CACHE = "results/attn_samples.pkl"

# =======================================
# Helper functions for SVA evaluation.
# =======================================
def get_logits(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

def compute_logit_diff(logits, correct_word: str, incorrect_word: str):
    correct_id = tokenizer.encode(correct_word, add_prefix_space=True)[0]
    incorrect_id = tokenizer.encode(incorrect_word, add_prefix_space=True)[0]
    logit_correct = logits[0, -1, correct_id].item()
    logit_incorrect = logits[0, -1, incorrect_id].item()
    return logit_correct - logit_incorrect

# =======================================
# PATH PATCHING FUNCTIONS (for attention heads)
# =======================================
def path_patch_head(layer_idx: int, head_idx: int, prompt_orig: str, prompt_patch: str):
    """
    For the given attention head (layer_idx, head_idx), capture its activation on prompt_patch,
    then run a forward pass on prompt_orig with that head’s output replaced.
    Returns the resulting logits.
    """
    # Step 1: Capture activation.
    inputs_patch = tokenizer(prompt_patch, return_tensors="pt")
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

    # Step 2: Replace activation.
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
    with torch.no_grad():
        outputs_orig = model(**inputs_orig)
    handle.remove()
    return outputs_orig.logits

def get_logit_diff_path_patch(layer_idx: int, head_idx: int, prompt: str, prompt_patch: str,
                              correct_word: str, incorrect_word: str):
    patched_logits = path_patch_head(layer_idx, head_idx, prompt, prompt_patch)
    return compute_logit_diff(patched_logits, correct_word, incorrect_word)

# =======================================
# Counterfactual Generator (flip subject's plurality for SVA)
# =======================================
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
    if example["is_negated"]:
        aux = "did not " if example["tense"] == "past" else ("do not " if new_is_plural else "does not ")
    else:
        aux = ""
    new_prompt = f"{example['prefix']} {new_subject} {aux}".strip()
    new_example = example.copy()
    new_example["prompt"] = new_prompt
    new_example["is_plural"] = new_is_plural
    new_example["subject"] = new_subject
    new_example["correct_verb"], new_example["incorrect_verb"] = example["incorrect_verb"], example["correct_verb"]
    return new_example

# =======================================
# ALTERNATIVE ABLATION METHODS for Attention Heads
# =======================================
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
    mean_act = cat_act.mean(dim=0)  # shape: (head_dim,)
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

# =======================================
# Knockout Function for Attention Heads with Multiple Ablation Methods.
# =======================================
def knockout_attn_heads_method(heads: list, prompt: str, method="zero", attn_means=None, attn_samples=None):
    inputs = tokenizer(prompt, return_tensors="pt")
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
        elif method == "resample":
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
    return outputs.logits

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

# =======================================
# Caching Functions.
# =======================================
def save_results(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_results(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

# =======================================
# Compute (or load) attention head mean activations and samples.
# =======================================
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

# =======================================
# Minimal Circuit Search with Beam Search for Attention Heads.
# =======================================
def find_minimal_attn_circuit_beam(sorted_heads, dataset, threshold=0.1, beam_size=10, num_samples=50, method="zero"):
    # Get baseline performance.
    full_eval = evaluate_attn_circuit([], dataset, num_samples=num_samples, method=method,
                                      attn_means=attn_means, attn_samples=attn_samples)
    F_full = full_eval["F_full"]
    beam = []
    best_candidate = None
    best_diff = float('inf')
    # Initialize beam with each single head candidate.
    for head in tqdm(sorted_heads, desc="Evaluating single head candidates"):
        candidate = [head]
        eval_candidate = evaluate_attn_circuit(candidate, dataset, num_samples=num_samples, method=method,
                                               attn_means=attn_means, attn_samples=attn_samples)
        diff = abs(eval_candidate["F_full"] - eval_candidate["F_circuit"]) / eval_candidate["F_full"]
        beam.append((candidate, diff))
        if diff < best_diff:
            best_candidate = candidate
            best_diff = diff
    beam = sorted(beam, key=lambda x: x[1])[:beam_size]
    improved = True
    while improved:
        new_beam = []
        improved = False
        for candidate, cand_diff in tqdm(beam, desc="Exploring beam candidates"):
            for head in sorted_heads:
                if head in candidate:
                    continue
                new_candidate = candidate + [head]
                eval_new = evaluate_attn_circuit(new_candidate, dataset, num_samples=num_samples, method=method,
                                                 attn_means=attn_means, attn_samples=attn_samples)
                diff = abs(eval_new["F_full"] - eval_new["F_circuit"]) / eval_new["F_full"]
                new_beam.append((new_candidate, diff))
                if diff < best_diff:
                    best_candidate = new_candidate
                    best_diff = diff
                    improved = True
        if new_beam:
            beam = sorted(new_beam, key=lambda x: x[1])[:beam_size]
        if best_diff < threshold:
            break
    final_eval = evaluate_attn_circuit(best_candidate, dataset, num_samples=num_samples, method=method,
                                       attn_means=attn_means, attn_samples=attn_samples)
    return best_candidate, final_eval

# Load or compute head effects from path patching.
if os.path.exists(HEAD_EFFECT_CACHE):
    with open(HEAD_EFFECT_CACHE, "rb") as f:
        head_effect_results = pickle.load(f)
else:
    def analyze_head_effects(dataset: list, num_samples: int = 50):
        sample_examples = random.sample(dataset, num_samples)
        num_layers = len(model.transformer.h)
        num_heads = model.config.n_head
        head_effects = {(layer, head): [] for layer in range(num_layers) for head in range(num_heads)}
        for ex in tqdm(sample_examples, desc="Analyzing head effects"):
            prompt_orig = ex["prompt"]
            baseline_logits = get_logits(prompt_orig)
            baseline_diff = compute_logit_diff(baseline_logits, ex["correct_verb"], ex["incorrect_verb"])
            ex_cf = generate_counterfactual(ex)
            prompt_cf = ex_cf["prompt"]
            for layer in range(num_layers):
                for head in range(num_heads):
                    patched_diff = get_logit_diff_path_patch(layer, head, prompt_orig, prompt_cf,
                                                             ex["correct_verb"], ex["incorrect_verb"])
                    effect = baseline_diff - patched_diff
                    head_effects[(layer, head)].append(effect)
        results = []
        for (layer, head), effects in head_effects.items():
            avg_effect = sum(effects) / len(effects)
            results.append({"layer": layer, "head": head, "avg_effect": avg_effect})
        return results
    head_effect_results = analyze_head_effects(dataset, num_samples=50)
    with open(HEAD_EFFECT_CACHE, "wb") as f:
        pickle.dump(head_effect_results, f)

df_effects = pd.DataFrame(head_effect_results)
df_effects["abs_effect"] = df_effects["avg_effect"].abs()
df_effects_sorted = df_effects.sort_values("abs_effect", ascending=False)
print("=== Path Patching Analysis Results (per head) ===")
print(df_effects_sorted)

# ----------------------------
# Generate and save a heatmap of attention head effects.
# ----------------------------
heatmap_data = df_effects.pivot(index='layer', columns='head', values='avg_effect')
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Attention Head Average Effects")
plt.xlabel("Head")
plt.ylabel("Layer")
plt.xticks(range(heatmap_data.shape[1]), heatmap_data.columns)
plt.yticks(range(heatmap_data.shape[0]), heatmap_data.index)
plt.savefig("attn_heatmap.png")
plt.close()
print("Saved heatmap as attn_heatmap.png")
# ----------------------------

sorted_candidate_heads = [(int(row["layer"]), int(row["head"])) for _, row in df_effects_sorted.iterrows()]

# Now, for each ablation method, run the minimal circuit search and save the result.
methods = ["zero", "mean", "resample"]
minimal_results = {}
for method in methods:
    print("\nSearching minimal circuit using ablation method:", method)
    if method == "zero":
        cache_file = MINIMAL_CIRCUIT_ZERO_CACHE
    elif method == "mean":
        cache_file = MINIMAL_CIRCUIT_MEAN_CACHE
    elif method == "resample":
        cache_file = MINIMAL_CIRCUIT_RESAMPLE_CACHE

    cached_result = load_results(cache_file)
    if cached_result is not None:
        minimal_circuit, minimal_eval = cached_result
        print(f"Loaded cached result for method '{method}'")
    else:
        minimal_circuit, minimal_eval = find_minimal_attn_circuit_beam(
            sorted_candidate_heads, dataset, threshold=0.1, beam_size=10, num_samples=50, method=method
        )
        save_results(cache_file, (minimal_circuit, minimal_eval))
    minimal_results[method] = (minimal_circuit, minimal_eval)
    print(f"\nMinimal circuit (attention heads) found using method '{method}':")
    print(minimal_circuit)
    print("Circuit evaluation:", minimal_eval)

# Optionally, save all minimal circuit results together.
with open("all_minimal_circuits.pkl", "wb") as f:
    pickle.dump(minimal_results, f)
print("Saved all minimal circuit results as all_minimal_circuits.pkl")