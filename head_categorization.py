import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer, GPT2Model
import os

# Directory to save plots
os.makedirs("attention_plots", exist_ok=True)

# List of heads identified in the circuit
circuit_heads = [(11, 6), (0, 4), (11, 4), (0, 8), (11, 7), (2, 6), (1, 0), (2, 1), (1, 1), (6, 0), (10, 0), (9, 4)]

# Prompts to visualize
prompts = ["Alice walk", "Alice walks"]

# Function to visualize and save attention heatmap
def visualize_and_save_attention(model_name, prompt, layer, head):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**inputs)

    attention = outputs.attentions[layer][0, head].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = len(tokens)

    plt.figure(figsize=(8, 6))
    sns.heatmap(attention[:seq_len, :seq_len], 
                xticklabels=tokens, 
                yticklabels=tokens, 
                cmap="viridis",
                square=True,
                annot=True, fmt=".2f")
    plt.title(f"GPT-2 Attention\nPrompt: '{prompt}'\nLayer {layer}, Head {head}")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    safe_prompt = prompt.replace(" ", "_")
    filename = f"attention_plots/{safe_prompt}_layer{layer}_head{head}.png"
    plt.savefig(filename)
    plt.close()

# Main loop to generate all plots
if __name__ == "__main__":
    model_name = "gpt2"

    for prompt in prompts:
        for layer, head in circuit_heads:
            visualize_and_save_attention(model_name, prompt, layer, head)
            print(f"Saved plot for prompt='{prompt}', layer={layer}, head={head}")
