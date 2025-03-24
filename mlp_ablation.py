import random
import torch
import pandas as pd
import pickle
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataset import generate_dataset_per_permutation

# attempted, not really successful

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

dataset = generate_dataset_per_permutation(samples_per_perm=100)

MLP_MEAN_CACHE = "results/mlp_mean_activations.pkl"
MLP_SAMPLES_CACHE = "results/mlp_activation_samples.pkl"
MLP_ABLATION_RESULTS_CACHE = "results/mlp_ablation_results.pkl"

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


def compute_mlp_mean_activation(layer_idx: int, dataset, num_samples: int = 50):
    """
    Compute the mean activation for the MLP at a given layer over a sample of examples.
    Returns a tensor of shape (hidden_size,).
    """
    activations = []
    sample_examples = random.sample(dataset, num_samples)
    
    def capture_hook(module, input, output):
        # output is a tensor of shape (batch, seq_len, hidden_size)
        activations.append(output.detach().cpu())
    
    handle = model.transformer.h[layer_idx].mlp.register_forward_hook(capture_hook)
    for ex in sample_examples:
        prompt = ex["prompt"]
        _ = get_logits(prompt)
    handle.remove()
    
    # Concatenate along batch and sequence dimensions.
    act_tensor = torch.cat(activations, dim=0)  # shape: (N, hidden_size) or (batch*seq_len, hidden_size)
    mean_act = act_tensor.mean(dim=0)  # shape: (hidden_size,)
    return mean_act

def compute_mlp_activation_samples(layer_idx: int, dataset, num_samples: int = 50, tokens_per_example: int = 5):
    """
    Collect a list of activation samples from the MLP at layer_idx.
    From each sampled example, randomly pick `tokens_per_example` token positions.
    Returns a list of tensors of shape (hidden_size,).
    """
    samples = []
    sample_examples = random.sample(dataset, num_samples)
    
    def capture_hook(module, input, output):
        # output shape: (batch, seq_len, hidden_size)
        # We'll select a few random token positions from the first (and only) batch.
        out_cpu = output.detach().cpu()[0]  # shape: (seq_len, hidden_size)
        seq_len = out_cpu.shape[0]
        indices = random.sample(range(seq_len), min(tokens_per_example, seq_len))
        for i in indices:
            samples.append(out_cpu[i])
    
    handle = model.transformer.h[layer_idx].mlp.register_forward_hook(capture_hook)
    for ex in sample_examples:
        prompt = ex["prompt"]
        _ = get_logits(prompt)
    handle.remove()
    return samples


def mlp_zero_hook_factory(layer_idx: int):
    """Zero ablation: replace MLP output with zeros."""
    def hook(module, input, output):
        return torch.zeros_like(output)
    return hook

def mlp_mean_hook_factory(layer_idx: int, mean_activation: torch.Tensor):
    """
    Mean ablation: replace MLP output with the mean activation (expanded to output shape).
    Assumes mean_activation has shape (hidden_size,).
    """
    def hook(module, input, output):
        batch, seq_len, hidden_size = output.shape
        # Expand mean_activation to shape (batch, seq_len, hidden_size)
        mean_expanded = mean_activation.view(1, 1, hidden_size).expand(batch, seq_len, hidden_size)
        return mean_expanded
    return hook

def mlp_resample_hook_factory(layer_idx: int, sample_list: list):
    """
    Resampling ablation: replace MLP output with a randomly chosen activation sample
    (expanded to match the output shape).
    Each sample in sample_list has shape (hidden_size,).
    """
    def hook(module, input, output):
        batch, seq_len, hidden_size = output.shape
        # Randomly choose one sample from sample_list.
        sample = random.choice(sample_list)
        sample_expanded = sample.view(1, 1, hidden_size).expand(batch, seq_len, hidden_size)
        return sample_expanded
    return hook

def load_or_compute_mlp_means(num_layers: int, dataset, num_samples=50):
    means = {}
    cached = load_results(MLP_MEAN_CACHE)
    if cached is not None:
        return cached
    for layer in range(num_layers):
        means[layer] = compute_mlp_mean_activation(layer, dataset, num_samples=num_samples)
        print(f"Computed mean activation for MLP layer {layer}.")
    save_results(MLP_MEAN_CACHE, means)
    return means

def load_or_compute_mlp_samples(num_layers: int, dataset, num_samples=50, tokens_per_example=5):
    samples = {}
    cached = load_results(MLP_SAMPLES_CACHE)
    if cached is not None:
        return cached
    for layer in range(num_layers):
        samples[layer] = compute_mlp_activation_samples(layer, dataset, num_samples=num_samples, tokens_per_example=tokens_per_example)
        print(f"Collected {len(samples[layer])} samples for MLP layer {layer}.")
    save_results(MLP_SAMPLES_CACHE, samples)
    return samples

def save_results(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_results(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    return None

def analyze_mlp_ablation(dataset: list, num_samples: int = 50):
    num_layers = model.config.n_layer
    results = []
    
    # Precompute (or load) mean activations and resampling samples.
    mlp_means = load_or_compute_mlp_means(num_layers, dataset, num_samples=num_samples)
    mlp_samples = load_or_compute_mlp_samples(num_layers, dataset, num_samples=num_samples, tokens_per_example=5)
    
    for layer in range(num_layers):
        effects_zero = []
        effects_mean = []
        effects_resample = []
        sample_examples = random.sample(dataset, num_samples)
        for ex in sample_examples:
            prompt = ex["prompt"]
            baseline_logits = get_logits(prompt)
            baseline_diff = compute_logit_diff(baseline_logits, ex["correct_verb"], ex["incorrect_verb"])
            
            # Zero Ablation
            handle_zero = model.transformer.h[layer].mlp.register_forward_hook(mlp_zero_hook_factory(layer))
            with torch.no_grad():
                logits_zero = get_logits(prompt)
            handle_zero.remove()
            diff_zero = compute_logit_diff(logits_zero, ex["correct_verb"], ex["incorrect_verb"])
            effects_zero.append(baseline_diff - diff_zero)
            
            # Mean Ablation
            handle_mean = model.transformer.h[layer].mlp.register_forward_hook(mlp_mean_hook_factory(layer, mlp_means[layer]))
            with torch.no_grad():
                logits_mean = get_logits(prompt)
            handle_mean.remove()
            diff_mean = compute_logit_diff(logits_mean, ex["correct_verb"], ex["incorrect_verb"])
            effects_mean.append(baseline_diff - diff_mean)
            
            # Resampling Ablation
            handle_resample = model.transformer.h[layer].mlp.register_forward_hook(mlp_resample_hook_factory(layer, mlp_samples[layer]))
            with torch.no_grad():
                logits_resample = get_logits(prompt)
            handle_resample.remove()
            diff_resample = compute_logit_diff(logits_resample, ex["correct_verb"], ex["incorrect_verb"])
            effects_resample.append(baseline_diff - diff_resample)
        
        avg_zero = sum(effects_zero) / len(effects_zero)
        avg_mean = sum(effects_mean) / len(effects_mean)
        avg_resample = sum(effects_resample) / len(effects_resample)
        results.append({"layer": layer,
                        "effect_zero": avg_zero,
                        "effect_mean": avg_mean,
                        "effect_resample": avg_resample})
        print(f"Layer {layer}: Zero={avg_zero:.4f}, Mean={avg_mean:.4f}, Resample={avg_resample:.4f}")
    return pd.DataFrame(results)

mlp_ablation_df = load_results(MLP_ABLATION_RESULTS_CACHE)
if mlp_ablation_df is None:
    mlp_ablation_df = analyze_mlp_ablation(dataset, num_samples=50)
    save_results(MLP_ABLATION_RESULTS_CACHE, mlp_ablation_df)
print("\n=== MLP Ablation Analysis Results ===")
print(mlp_ablation_df)
