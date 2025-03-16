import random
import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import generate_dataset_per_permutation
# -------------------------------
# Load GPT-2 small and set to eval mode.
# -------------------------------
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# -------------------------------
# Generate dataset with 100 samples per permutation.
# -------------------------------
dataset = generate_dataset_per_permutation(samples_per_perm=100)

# -------------------------------
# Helper functions
# -------------------------------
def get_logits(prompt: str):
    """Return model logits for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

def compute_logit_diff(logits, correct_word: str, incorrect_word: str):
    """
    Given logits, compute the logit difference for the next-token prediction.
    (Difference between the logit for the correct token and the incorrect token.)
    """
    correct_id = tokenizer.encode(correct_word, add_prefix_space=True)[0]
    incorrect_id = tokenizer.encode(incorrect_word, add_prefix_space=True)[0]
    logit_correct = logits[0, -1, correct_id].item()
    logit_incorrect = logits[0, -1, incorrect_id].item()
    return logit_correct - logit_incorrect

# -------------------------------
# Path patching functions.
# -------------------------------
def path_patch_head(layer_idx: int, head_idx: int, prompt_orig: str, prompt_patch: str):
    """
    For the given layer and head, run two forward passes:
      1. On prompt_patch to capture the activation for that head.
      2. On prompt_orig while replacing that head's output with the captured activation.
    Returns the logits from the patched forward pass.
    """
    # --- Step 1: Capture the patch activation from prompt_patch.
    inputs_patch = tokenizer(prompt_patch, return_tensors="pt")
    patch_activation = None

    def capture_hook(module, input, output):
        nonlocal patch_activation
        # If output is a tuple, take the first element.
        if isinstance(output, tuple):
            output = output[0]
        head_dim = output.shape[-1] // module.num_heads
        patch_activation_local = output[..., head_idx * head_dim:(head_idx + 1) * head_dim].clone()
        # Ensure a batch dimension is present.
        if patch_activation_local.dim() == 2:
            patch_activation_local = patch_activation_local.unsqueeze(0)
        patch_activation = patch_activation_local

    hook_handle = model.transformer.h[layer_idx].attn.register_forward_hook(capture_hook)
    _ = model(**inputs_patch)
    hook_handle.remove()

    if patch_activation is None:
        raise RuntimeError("Failed to capture patch activation.")

    # --- Step 2: Replace the head's output in a forward pass on prompt_orig.
    def replace_hook(module, input, output):
        # If output is a tuple, extract the tensor part.
        tuple_out = False
        if isinstance(output, tuple):
            output_tensor, rest = output[0], output[1:]
            tuple_out = True
        else:
            output_tensor = output

        head_dim = output_tensor.shape[-1] // module.num_heads
        patched = output_tensor.clone()
        batch_size, seq_len, _ = patched.shape

        # Ensure patch_activation has a batch dimension.
        if patch_activation.dim() == 2:
            patch_act = patch_activation.unsqueeze(0)
        else:
            patch_act = patch_activation

        patch_seq_len = patch_act.shape[1]
        min_seq_len = min(seq_len, patch_seq_len)
        # Replace only for the first min_seq_len tokens.
        patched[:, :min_seq_len, head_idx * head_dim:(head_idx + 1) * head_dim] = \
            patch_act[:, :min_seq_len, :]
        if tuple_out:
            return (patched,) + rest
        return patched

    hook_handle = model.transformer.h[layer_idx].attn.register_forward_hook(replace_hook)
    inputs_orig = tokenizer(prompt_orig, return_tensors="pt")
    with torch.no_grad():
        outputs_orig = model(**inputs_orig)
    hook_handle.remove()

    return outputs_orig.logits


def get_logit_diff_path_patch(layer_idx: int, head_idx: int, prompt: str, prompt_patch: str,
                              correct_word: str, incorrect_word: str):
    """Run path patching for a given head and return the computed logit difference."""
    patched_logits = path_patch_head(layer_idx, head_idx, prompt, prompt_patch)
    return compute_logit_diff(patched_logits, correct_word, incorrect_word)

# -------------------------------
# Counterfactual generator.
# For subject-verb agreement, we flip the subject's plurality.
# -------------------------------
# (Make sure that these name lists match those used in dataset generation.)
singular_names = [
    "Alice", "Bob", "Charlie", "Dana", "Eve", "Frank", "Grace", "Hank",
    "Ivy", "Jack", "Kara", "Leo", "Mia", "Nina", "Oscar", "Paul", "Quinn", "Rose", "Sam", "Tina"
]
plural_names = [f"{n1} and {n2}" for n1 in singular_names for n2 in singular_names if n1 != n2]
singular_pronouns = ["She", "He"]
plural_pronouns = ["They"]

def generate_counterfactual(example: dict) -> dict:
    """
    Create a counterfactual version of an example by flipping the subject's plurality.
    This effectively swaps the correct and incorrect verb forms.
    """
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
    # Swap correct and incorrect verbs.
    new_example["correct_verb"], new_example["incorrect_verb"] = example["incorrect_verb"], example["correct_verb"]
    return new_example

# -------------------------------
# Systematic analysis over the dataset via path patching.
# -------------------------------
def analyze_head_effects(dataset: list, num_samples: int = 50):
    """
    For each attention head (all layers and heads in GPT-2 small), compute the average effect on the
    logit difference when patching in the activation from a counterfactual example.
    Effect = (baseline logit diff) - (patched logit diff)
    Returns a list of dictionaries for each head: {layer, head, avg_effect}.
    """
    sample_examples = random.sample(dataset, num_samples)
    num_layers = len(model.transformer.h)
    num_heads = model.config.n_head
    head_effects = {(layer, head): [] for layer in range(num_layers) for head in range(num_heads)}
    
    for ex in sample_examples:
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

# -------------------------------
# Knockout analysis: zero-out selected heads.
# -------------------------------
def zero_hook_factory(head_idx: int):
    """
    Returns a hook function that zeroes out the output for the specified head.
    Handles the case when the module's output is a tuple.
    """
    def hook(module, input, output):
        # Check if output is a tuple
        if isinstance(output, tuple):
            output_tensor, rest = output[0], output[1:]
            head_dim = output_tensor.shape[-1] // module.num_heads
            patched = output_tensor.clone()
            patched[..., head_idx * head_dim:(head_idx+1)*head_dim] = 0
            return (patched,) + rest
        else:
            head_dim = output.shape[-1] // module.num_heads
            patched = output.clone()
            patched[..., head_idx * head_dim:(head_idx+1)*head_dim] = 0
            return patched
    return hook

def knockout_heads(heads: list, prompt: str):
    """
    Given a list of heads (tuples (layer, head)), run a forward pass on the prompt with those heads zeroed out.
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

def analyze_knockout_effect(heads: list, dataset: list, num_samples: int = 50):
    """
    For a given set of heads, compute the average effect on the logit difference over a random sample
    from the dataset.
    Effect = baseline logit diff - knockout logit diff.
    """
    sample_examples = random.sample(dataset, num_samples)
    effects = []
    for ex in sample_examples:
        baseline_logits = get_logits(ex["prompt"])
        baseline_diff = compute_logit_diff(baseline_logits, ex["correct_verb"], ex["incorrect_verb"])
        knocked_logits = knockout_heads(heads, ex["prompt"])
        knocked_diff = compute_logit_diff(knocked_logits, ex["correct_verb"], ex["incorrect_verb"])
        effects.append(baseline_diff - knocked_diff)
    avg_effect = sum(effects) / len(effects)
    return avg_effect

# -------------------------------
# Full circuit evaluation.
# Here we define a "circuit" as a set of candidate heads.
# We simulate circuit-only behavior by knocking out all heads not in the circuit.
# -------------------------------
def evaluate_circuit(circuit_heads: list, dataset: list, num_samples: int = 50):
    """
    Evaluate a circuit by comparing:
      - F_full: average logit difference of the full model.
      - F_circuit: average logit difference when only the heads in the circuit are active.
    Returns a dict with F_full, F_circuit, and a faithfulness score.
    """
    sample_examples = random.sample(dataset, num_samples)
    full_diffs = []
    circuit_diffs = []
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    all_heads = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
    
    for ex in sample_examples:
        baseline_logits = get_logits(ex["prompt"])
        full_diff = compute_logit_diff(baseline_logits, ex["correct_verb"], ex["incorrect_verb"])
        full_diffs.append(full_diff)
        
        # Knock out all heads not in the circuit.
        heads_to_knockout = [h for h in all_heads if h not in circuit_heads]
        circuit_logits = knockout_heads(heads_to_knockout, ex["prompt"])
        circuit_diff = compute_logit_diff(circuit_logits, ex["correct_verb"], ex["incorrect_verb"])
        circuit_diffs.append(circuit_diff)
    
    F_full = sum(full_diffs) / len(full_diffs)
    F_circuit = sum(circuit_diffs) / len(circuit_diffs)
    faithfulness = abs(F_full - F_circuit) / F_full if F_full != 0 else None
    return {"F_full": F_full, "F_circuit": F_circuit, "faithfulness": faithfulness}

# -------------------------------
# Run the analyses.
# -------------------------------
# (Assuming your dataset variable is already defined.)
head_effect_results = analyze_head_effects(dataset, num_samples=50)
df_effects = pd.DataFrame(head_effect_results)
df_effects["abs_effect"] = df_effects["avg_effect"].abs()
df_effects_sorted = df_effects.sort_values("abs_effect", ascending=False)
print("=== Path Patching Analysis Results (per head) ===")
print(df_effects_sorted)

# For example, select the top 5 heads (by absolute average effect) as our candidate circuit.
top5 = df_effects_sorted.head(5)[["layer", "head"]].to_records(index=False)
top5_list = [(int(x[0]), int(x[1])) for x in top5]
knockout_effect = analyze_knockout_effect(top5_list, dataset, num_samples=50)
print("\nAverage knockout effect for top 5 candidate heads:", knockout_effect)

circuit_eval = evaluate_circuit(top5_list, dataset, num_samples=50)
print("\nCircuit evaluation (top 5 heads):")
print(circuit_eval)
