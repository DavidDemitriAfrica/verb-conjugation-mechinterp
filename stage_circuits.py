from itertools import product
import random
import math
import torch
import pandas as pd
import pickle
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    """
    Compute the difference between the summed logits for all tokens corresponding
    to correct_word and incorrect_word.
    """
    correct_ids = tokenizer.encode(correct_word, add_prefix_space=True)
    incorrect_ids = tokenizer.encode(incorrect_word, add_prefix_space=True)
    logit_correct = sum(logits[0, -1, token_id].item() for token_id in correct_ids)
    logit_incorrect = sum(logits[0, -1, token_id].item() for token_id in incorrect_ids)
    return logit_correct - logit_incorrect

def path_patch_head(layer_idx: int, head_idx: int, prompt_orig: str, prompt_patch: str):
    # Capture activation from the patch prompt.
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
    new_example["correct_verb"], new_example["incorrect_verb"] = example["correct_verb"], example["incorrect_verb"]
    return new_example

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
        elif method in ["resample", "interchange"]:
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
    # Use min(num_samples, len(dataset)) to avoid sampling more than available.
    sample_examples = random.sample(dataset, min(num_samples, len(dataset)))
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

def evaluate_full_model(dataset):
    predictions = []
    golds = []
    for ex in tqdm(dataset, desc="Evaluating full model accuracy"):
        logits = get_logits(ex["prompt"])
        logit_diff = compute_logit_diff(logits, ex["correct_verb"], ex["incorrect_verb"])
        pred = 1 if logit_diff > 0 else 0
        predictions.append(pred)
        golds.append(1)
    return (accuracy_score(golds, predictions),
            precision_score(golds, predictions, zero_division=0),
            recall_score(golds, predictions, zero_division=0),
            f1_score(golds, predictions, zero_division=0))

def save_results(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_results(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def filter_dataset_by_setting(dataset, setting):
    """
    Filter the dataset based on a given setting.
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
        # Use the correct key "use_irregular"
        return [ex for ex in dataset if ex.get("use_irregular", False) is True]
    elif setting == "complex":
        # Maximally complex: all conditions turned on.
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

def iterative_search_stage(initial_circuit, candidate_heads, dataset, method, attn_means, attn_samples, num_samples=50):
    """
    Iteratively scan candidate heads over the provided (filtered) dataset.
    If adding a candidate head improves the circuit's F_circuit performance,
    immediately add it (printing the increase in accuracy) and restart scanning.
    """
    circuit = initial_circuit.copy() if initial_circuit is not None else []
    improvement = True
    while improvement:
        improvement = False
        current_eval = evaluate_attn_circuit(circuit, dataset, num_samples=num_samples, method=method,
                                               attn_means=attn_means, attn_samples=attn_samples)
        current_perf = current_eval["F_circuit"]
        # Use a tqdm loading bar for candidate scanning.
        for head in tqdm(candidate_heads, desc="Scanning candidate heads", leave=False):
            if head in circuit:
                continue
            candidate_circuit = circuit + [head]
            candidate_eval = evaluate_attn_circuit(candidate_circuit, dataset, num_samples=num_samples, method=method,
                                                   attn_means=attn_means, attn_samples=attn_samples)
            candidate_perf = candidate_eval["F_circuit"]
            if candidate_perf > current_perf:
                improvement_value = candidate_perf - current_perf
                circuit.append(head)
                improvement = True
                print(f"  Added head {head}: F_circuit improved from {current_perf:.4f} to {candidate_perf:.4f} (increase: {improvement_value:.4f})")
                break  # Immediately update circuit and restart scanning.
    final_eval = evaluate_attn_circuit(circuit, dataset, num_samples=num_samples, method=method,
                                       attn_means=attn_means, attn_samples=attn_samples)
    return circuit, final_eval

def iterative_search_for_setting(setting, candidate_heads, dataset, method, attn_means, attn_samples, num_samples=50, initial_circuit=None):
    """
    Filter the dataset for a given setting, then run iterative search (starting with initial_circuit if provided,
    otherwise an empty circuit) to pick heads that improve performance for that setting.
    Returns the circuit and evaluation metrics.
    """
    filtered_dataset = filter_dataset_by_setting(dataset, setting)
    print(f"\n--- Running iterative search for setting '{setting}' ---")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    if len(filtered_dataset) < num_samples:
        print(f"Warning: Filtered dataset for setting '{setting}' is smaller than num_samples ({num_samples}). Using all available samples.")
    circuit, eval_metrics = iterative_search_stage(initial_circuit, candidate_heads, filtered_dataset, method,
                                                     attn_means, attn_samples, num_samples=num_samples)
    print(f"Result for setting '{setting}': Circuit = {circuit}")
    print(f"Evaluation = {eval_metrics}")
    return circuit, eval_metrics

def freeze_circuit(circuit):
    """
    Permanently modify the model by registering forward hooks on all heads NOT in 'circuit'
    so that their activations are zeroed out.
    These hooks are not removed, effectively freezing the model to use only the heads in 'circuit'.
    """
    for layer in range(model.config.n_layer):
        for head in range(model.config.n_head):
            if (layer, head) not in circuit:
                model.transformer.h[layer].attn.register_forward_hook(attn_zero_hook_factory(head))
    return model

if __name__ == "__main__":
    # Load or generate dataset.
    from dataset import generate_dataset_per_permutation
    dataset = generate_dataset_per_permutation(samples_per_perm=10)

    # Define cache file paths.
    ATTN_MEANS_CACHE = "results/attn_means.pkl"
    ATTN_SAMPLES_CACHE = "results/attn_samples.pkl"
    HEAD_EFFECT_CACHE = "results/attn_head_effect_results.pkl"

    # Load or compute attn_means.
    if os.path.exists(ATTN_MEANS_CACHE):
        with open(ATTN_MEANS_CACHE, "rb") as f:
            attn_means = pickle.load(f)
    else:
        def compute_all_attn_means(dataset, num_samples=50):
            num_layers = len(model.transformer.h)
            num_heads = model.config.n_head
            attn_means = {layer: {} for layer in range(num_layers)}
            for layer in tqdm(range(num_layers), desc="Computing attn means per layer"):
                for head in tqdm(range(num_heads), desc=f"Layer {layer} heads", leave=False):
                    mean_act = compute_attn_head_mean_activation(layer, head, dataset, num_samples)
                    attn_means[layer][head] = mean_act
            return attn_means
        attn_means = compute_all_attn_means(dataset, num_samples=50)
        save_results(ATTN_MEANS_CACHE, attn_means)

    # Load or compute attn_samples.
    if os.path.exists(ATTN_SAMPLES_CACHE):
        with open(ATTN_SAMPLES_CACHE, "rb") as f:
            attn_samples = pickle.load(f)
    else:
        def compute_all_attn_samples(dataset, num_samples=50, tokens_per_example=5):
            num_layers = len(model.transformer.h)
            num_heads = model.config.n_head
            attn_samples = {layer: {} for layer in range(num_layers)}
            for layer in tqdm(range(num_layers), desc="Collecting attn samples per layer"):
                for head in tqdm(range(num_heads), desc=f"Layer {layer} heads", leave=False):
                    samples = compute_attn_head_samples(layer, head, dataset, num_samples, tokens_per_example)
                    attn_samples[layer][head] = samples
            return attn_samples
        attn_samples = compute_all_attn_samples(dataset, num_samples=50, tokens_per_example=5)
        save_results(ATTN_SAMPLES_CACHE, attn_samples)

    # Load or compute head effects (for candidate head ranking).
    if os.path.exists(HEAD_EFFECT_CACHE):
        with open(HEAD_EFFECT_CACHE, "rb") as f:
            head_effect_results = pickle.load(f)
    else:
        print("Computing head effects via path patching for each attention head...")
        head_effect_results = []
        sample_dataset = random.sample(dataset, min(20, len(dataset)))
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
                        orig_logits = get_logits(prompt_orig)
                        orig_logit_diff = compute_logit_diff(orig_logits, ex["correct_verb"], ex["incorrect_verb"])
                        patched_logits = path_patch_head(layer, head, prompt_orig, prompt_cf)
                        patched_logit_diff = compute_logit_diff(patched_logits, ex["correct_verb"], ex["incorrect_verb"])
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
    print("=== Path Patching Analysis Results (Top 20 Heads) ===")
    print(df_effects_sorted.head(20))
    
    # Get candidate heads sorted by effect.
    sorted_candidate_heads = [
        (int(row["layer"]), int(row["head"]))
        for _, row in df_effects_sorted.iterrows()
    ]

    # Define the settings to evaluate.
    settings = ["base", "plural", "negation", "prefix", "pronoun", "past_tense", "irregular", "complex"]

    # Check for cumulative results and load them if present.
    cumulative_results_file = "iterative_search_by_setting_results.pkl"
    if os.path.exists(cumulative_results_file):
        cumulative_data = load_results(cumulative_results_file)
        setting_results = cumulative_data.get("setting_results", {})
        head_usage = cumulative_data.get("head_usage", {})
        print(f"Loaded cumulative results from '{cumulative_results_file}'")
    else:
        setting_results = {}
        head_usage = {}

    if "base" not in setting_results:
        base_circuit, base_eval = iterative_search_for_setting(
            "base", candidate_heads=sorted_candidate_heads, dataset=dataset,
            method="resample", attn_means=attn_means, attn_samples=attn_samples, num_samples=50, initial_circuit=[]
        )
        setting_results["base"] = {"circuit": base_circuit, "eval": base_eval}
        for head in base_circuit:
            head_usage.setdefault(head, []).append("base")
        print(f"\nBase circuit: {base_circuit}")
        print(f"Base evaluation: {base_eval}")
        # Save cumulative results after base stage.
        save_results(cumulative_results_file, {"setting_results": setting_results, "head_usage": head_usage})
        print(f"Saved cumulative results after base stage to '{cumulative_results_file}'")
        # Freeze the base circuit in the model and save the modified model.
        model = freeze_circuit(base_circuit)
        torch.save(model.state_dict(), "model_base.pt")
        print("Saved base model with circuit frozen as 'model_base.pt'")
    else:
        base_circuit = setting_results["base"]["circuit"]
        print(f"Using previously saved base circuit: {base_circuit}")

    for s in settings:
        if s == "base":
            continue
        if s in setting_results:
            print(f"Setting '{s}' already completed. Skipping.")
            continue
        # Start each subsequent search from the base circuit.
        circuit, eval_metrics = iterative_search_for_setting(
            s, candidate_heads=sorted_candidate_heads, dataset=dataset,
            method="resample", attn_means=attn_means, attn_samples=attn_samples, num_samples=50,
            initial_circuit=base_circuit.copy()
        )
        setting_results[s] = {"circuit": circuit, "eval": eval_metrics}
        for head in circuit:
            head_usage.setdefault(head, []).append(s)
        # Save cumulative results after this stage.
        save_results(cumulative_results_file, {"setting_results": setting_results, "head_usage": head_usage})
        print(f"Saved cumulative results after stage '{s}' to '{cumulative_results_file}'")

    print("\n=== Summary of Head Usage Across Settings ===")
    for head, settings_used in head_usage.items():
        print(f"Head {head} was selected for settings: {settings_used}")

    # Save final cumulative results.
    save_results(cumulative_results_file, {"setting_results": setting_results, "head_usage": head_usage})
    print(f"\nFinal cumulative results saved as '{cumulative_results_file}'")
