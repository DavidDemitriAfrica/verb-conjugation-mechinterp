import random
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product
from dataset import generate_dataset_per_permutation
from utils import get_logit_diff, get_logits

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

dataset = generate_dataset_per_permutation(samples_per_perm=100)
singular_names = [
    "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Kara", "Leo", "Mia", "Nina", "Oscar", "Paul", "Quinn", "Rose", "Sam", "Tina"
]
plural_names = [f"{n1} and {n2}" for n1 in singular_names for n2 in singular_names if n1 != n2]
singular_pronouns = ["She", "He"]
plural_pronouns = ["They"]

# Define time-specific prefixes for each tense.
past_prefixes = ["Yesterday, ", "Last week, "]
present_prefixes = ["Today, ", "In the morning, ", "At night, "]

# Extended regular verb forms.
reg_present_singular = ["walks", "runs", "talks", "jumps", "sleeps", "writes", "reads", "sings", "dances"]
reg_present_plural   = ["walk", "run", "talk", "jump", "sleep", "write", "read", "sing", "dance"]
reg_past = ["walked", "ran", "talked", "jumped", "slept", "wrote", "read", "sang", "danced"]

# Irregular verbs with forms.
irregular_verbs = {
    "go": {"present_singular": "goes", "present_plural": "go", "past": "went"},
    "eat": {"present_singular": "eats", "present_plural": "eat", "past": "ate"},
    "have": {"present_singular": "has", "present_plural": "have", "past": "had"}
}

# All permutation combinations.
conditions = list(product([False, True],    # is_plural
                            [False, True],    # is_negated
                            [False, True],    # has_prefix
                            [False, True],    # is_pronoun
                            ["present", "past"],  # tense
                            [False, True]))   # use_irregular

# =============================================================================
# Path patching functions.
# =============================================================================
def path_patch_head(layer_idx, head_idx, prompt_orig, prompt_patch):
    """
    For the given layer and head index, run two forward passes:
      1. On prompt_patch to capture the activation for that head.
      2. On prompt_orig while replacing that head's output with the captured activation.
    Returns the logits from the patched forward pass.
    """
    # --- Step 1. Capture the patch activation from prompt_patch.
    inputs_patch = tokenizer(prompt_patch, return_tensors="pt")
    patch_activation = None

    def capture_hook(module, input, output):
        nonlocal patch_activation
        # output has shape (batch, seq_length, hidden_size)
        head_dim = output.shape[-1] // module.num_heads
        # Extract the slice corresponding to our head.
        patch_activation = output[..., head_idx * head_dim:(head_idx + 1) * head_dim].clone()

    hook_handle = model.transformer.h[layer_idx].attn.register_forward_hook(capture_hook)
    _ = model(**inputs_patch)
    hook_handle.remove()

    if patch_activation is None:
        raise RuntimeError("Failed to capture patch activation.")

    # --- Step 2. Replace the head's output in a forward pass on prompt_orig.
    def replace_hook(module, input, output):
        head_dim = output.shape[-1] // module.num_heads
        patched = output.clone()
        patched[..., head_idx * head_dim:(head_idx + 1) * head_dim] = patch_activation
        return patched

    hook_handle = model.transformer.h[layer_idx].attn.register_forward_hook(replace_hook)
    inputs_orig = tokenizer(prompt_orig, return_tensors="pt")
    with torch.no_grad():
        outputs_orig = model(**inputs_orig)
    hook_handle.remove()

    return outputs_orig.logits

def get_logit_diff_path_patch(layer_idx, head_idx, prompt, prompt_patch, correct_word, incorrect_word):
    """
    Compute the logit difference on `prompt` when patching the activation of a specific
    head (from prompt_patch) into the forward pass.
    """
    patched_logits = path_patch_head(layer_idx, head_idx, prompt, prompt_patch)
    return get_logit_diff(patched_logits, correct_word, incorrect_word)

# =============================================================================
# Define a counterfactual generator.
# For our subject-verb agreement task, we flip the subject number.
# =============================================================================
def generate_counterfactual(example):
    """
    Create a counterfactual version of an example by flipping the subject's plurality.
    (This in effect swaps the correct and incorrect verb forms.)
    """
    new_is_plural = not example["is_plural"]
    if example["is_pronoun"]:
        new_subject = random.choice(plural_pronouns) if new_is_plural else random.choice(singular_pronouns)
    else:
        new_subject = random.choice(plural_names) if new_is_plural else random.choice(singular_names)
    # Construct auxiliary (if negated) based on tense and new subject number.
    if example["is_negated"]:
        aux = "did not " if example["tense"] == "past" else ("do not " if new_is_plural else "does not ")
    else:
        aux = ""
    new_prompt = f"{example['prefix']} {new_subject} {aux}".strip()
    new_example = example.copy()
    new_example["prompt"] = new_prompt
    new_example["is_plural"] = new_is_plural
    new_example["subject"] = new_subject
    # Swap correct and incorrect verbs.
    new_example["correct_verb"], new_example["incorrect_verb"] = example["incorrect_verb"], example["correct_verb"]
    return new_example

# =============================================================================
# Systematic analysis over the dataset via path patching.
# =============================================================================
def analyze_head_effects(dataset, num_samples=50):
    """
    For each attention head (all layers and heads in GPT-2 small), compute the average effect on the
    logit difference when patching in the activation from a counterfactual example.
    Effect is defined as:
       effect = (baseline logit diff) - (patched logit diff)
    Returns a list of dictionaries for each head with keys: 'layer', 'head', 'avg_effect'.
    """
    sample_examples = random.sample(dataset, num_samples)
    # Initialize dictionary to collect effects per head.
    head_effects = {}
    num_layers = len(model.transformer.h)
    num_heads = model.config.n_head
    for layer in range(num_layers):
        for head in range(num_heads):
            head_effects[(layer, head)] = []
    
    for ex in sample_examples:
        prompt_orig = ex["prompt"]
        # Baseline performance on original prompt.
        baseline_logits = get_logits(prompt_orig)
        baseline_diff = get_logit_diff(baseline_logits, ex["correct_verb"], ex["incorrect_verb"])
        # Generate counterfactual by flipping subject number.
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

# Run the path patching analysis (using, e.g., 50 randomly sampled examples).
head_effect_results = analyze_head_effects(dataset, num_samples=50)
df_effects = pd.DataFrame(head_effect_results)
df_effects["abs_effect"] = df_effects["avg_effect"].abs()
df_effects_sorted = df_effects.sort_values("abs_effect", ascending=False)
print("=== Path Patching Analysis Results (per head) ===")
print(df_effects_sorted)

# =============================================================================
# Knockout analysis: zero-out selected heads.
# =============================================================================
def zero_hook_factory(head_idx):
    """
    Returns a hook that zeroes out the output of the specified head.
    """
    def hook(module, input, output):
        head_dim = output.shape[-1] // module.num_heads
        patched = output.clone()
        patched[..., head_idx * head_dim:(head_idx+1)*head_dim] = 0
        return patched
    return hook

def knockout_heads(heads, prompt):
    """
    Given a list of heads (each a tuple (layer, head)), run a forward pass on the prompt with
    those heads zeroed out.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    hook_handles = []
    for (layer, head) in heads:
        handle = model.transformer.h[layer].attn.register_forward_hook(zero_hook_factory(head))
        hook_handles.append(handle)
    with torch.no_grad():
        outputs = model(**inputs)
    for handle in hook_handles:
        handle.remove()
    return outputs.logits

def analyze_knockout_effect(heads, dataset, num_samples=50):
    """
    For a given set of heads, compute the average effect on logit difference (baseline minus
    knockout) over a random sample from the dataset.
    """
    sample_examples = random.sample(dataset, num_samples)
    effects = []
    for ex in sample_examples:
        baseline_logits = get_logits(ex["prompt"])
        baseline_diff = get_logit_diff(baseline_logits, ex["correct_verb"], ex["incorrect_verb"])
        knocked_logits = knockout_heads(heads, ex["prompt"])
        knocked_diff = get_logit_diff(knocked_logits, ex["correct_verb"], ex["incorrect_verb"])
        effects.append(baseline_diff - knocked_diff)
    avg_effect = sum(effects) / len(effects)
    return avg_effect

# For example, suppose we take the top 5 heads from our path patching analysis:
top5 = df_effects_sorted.head(5)[["layer", "head"]].to_records(index=False)
top5_list = [(int(x[0]), int(x[1])) for x in top5]
knockout_effect = analyze_knockout_effect(top5_list, dataset, num_samples=50)
print("\nAverage knockout effect for top 5 candidate heads:", knockout_effect)

# =============================================================================
# Full circuit evaluation (a skeleton example).
# Here we define a "circuit" as a set of candidate heads.
# We then measure the model's average logit difference on a set of examples when only the circuit is active.
# (This is done by knocking out all heads not in the circuit.)
# =============================================================================
def evaluate_circuit(circuit_heads, dataset, num_samples=50):
    """
    Evaluate a circuit (set of heads) by comparing:
      - F_full: average logit difference of the full model.
      - F_circuit: average logit difference when only the heads in the circuit are active.
    We simulate "circuit-only" behavior by zeroing out all heads not in the circuit.
    Returns a dictionary with F_full, F_circuit, and a simple faithfulness score.
    """
    sample_examples = random.sample(dataset, num_samples)
    full_diffs = []
    circuit_diffs = []
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    
    for ex in sample_examples:
        baseline_logits = get_logits(ex["prompt"])
        full_diff = get_logit_diff(baseline_logits, ex["correct_verb"], ex["incorrect_verb"])
        full_diffs.append(full_diff)
        
        # Knockout all heads not in circuit.
        heads_to_knockout = [h for h in all_heads if h not in circuit_heads]
        circuit_logits = knockout_heads(heads_to_knockout, ex["prompt"])
        circuit_diff = get_logit_diff(circuit_logits, ex["correct_verb"], ex["incorrect_verb"])
        circuit_diffs.append(circuit_diff)
    
    F_full = sum(full_diffs) / len(full_diffs)
    F_circuit = sum(circuit_diffs) / len(circuit_diffs)
    faithfulness = abs(F_full - F_circuit) / F_full
    return {"F_full": F_full, "F_circuit": F_circuit, "faithfulness": faithfulness}

# For instance, evaluate the circuit formed by our top 5 heads:
circuit_eval = evaluate_circuit(top5_list, dataset, num_samples=50)
print("\nCircuit evaluation (top 5 heads):")
print(circuit_eval)
