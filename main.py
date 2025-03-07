import random
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from tqdm import tqdm

# ---------------------------
# Helper functions.
# ---------------------------
def get_article(word):
    """Returns the appropriate indefinite article for a word."""
    return "an" if word[0].lower() in "aeiou" else "a"

def format_object_list(objects):
    """Formats a list of objects in natural language."""
    formatted = [f"{get_article(obj)} {obj}" for obj in objects]
    if len(formatted) == 1:
        return formatted[0]
    return ", ".join(formatted[:-1]) + ", and " + formatted[-1]

# Expanded vocabulary lists.
fruits = ["apple", "banana", "pear", "orange", "grape", "mango", "kiwi", "peach", "plum", "cherry"]
non_fruits = ["laptop", "book", "pen", "phone", "table", "chair", "clock", "bottle", "cup", "notebook"]
names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah", "Ivan", "Judy"]
places = ["school", "market", "park", "mall", "office", "restaurant", "beach", "library", "stadium", "cinema"]
times = ["this morning", "after work", "yesterday", "last night", "today"]

# Additional prefix and suffix variations.
prefix_options = [
    "Then, {time}",
    "Before going to class",
    "After visiting the {place}",
    "While at the {place}",
    "{time}"
]

suffix_options = [
    "The number of fruits I am holding is",
    "The number of fruits I had was",
    "In total, the number of fruits I held is"
]

# Templates for each setting.
templates = {
    "default": [
        "{prefix}, I picked up {objects}. {suffix} {count}.",
        "I picked up {objects}. {suffix} {count}."
    ],
    "negation": [
        "{prefix}, I picked up {positive_objects} but not {negative_objects}. {suffix} {pos_count}.",
        "I picked up {positive_objects} but did not pick up {negative_objects}. {suffix} {pos_count}."
    ],
    "object_classes": [
        "{prefix}, I picked up {fruit_objects} and {nonfruit_objects}. {suffix} {fruit_count}.",
        "I picked up {fruit_objects} along with {nonfruit_objects}. {suffix} {fruit_count}."
    ],
    "multiple_people": [
        "{prefix}, I picked up {my_objects}. {name} picked up {other_objects}. The number of fruits {name} had was {count}.",
        "I picked up {my_objects}. Meanwhile, {name} picked up {other_objects}. In total, the number of fruits {name} had was {count}."
    ]
}

def generate_counting_examples(seed=42, setting="default"):
    random.seed(seed)
    # Randomly choose a prefix and suffix variant.
    prefix_template = random.choice(prefix_options)
    suffix_template = random.choice(suffix_options)
    time_phrase = random.choice(times)
    place_phrase = random.choice(places)
    # Fill in {time} and {place} and ensure the result starts with a capital letter.
    prefix = prefix_template.format(time=time_phrase, place=place_phrase)
    prefix = prefix[0].upper() + prefix[1:]
    
    # Prepare output variables.
    sentence = ""
    answer = None
    
    if setting == "default":
        # Randomize number of fruits between 1 and 5.
        count = random.randint(1, 5)
        selected_fruits = random.sample(fruits, count)
        objects_str = format_object_list(selected_fruits)
        fruit_word = "fruit" if count == 1 else "fruits"
        template = random.choice(templates["default"])
        sentence = template.format(prefix=prefix, objects=objects_str, suffix=suffix_template,
                                   count=count, fruit_word=fruit_word)
        answer = count
    
    elif setting == "negation":
        # Randomly choose number of positive fruits and negated objects (1 to 3 each).
        pos_count = random.randint(1, 3)
        neg_count = random.randint(1, 3)
        pos_fruits = random.sample(fruits, pos_count)
        negative_items = random.sample(fruits + non_fruits, neg_count)
        pos_objects_str = format_object_list(pos_fruits)
        neg_objects_str = format_object_list(negative_items)
        fruit_word = "fruit" if pos_count == 1 else "fruits"
        template = random.choice(templates["negation"])
        sentence = template.format(prefix=prefix, positive_objects=pos_objects_str,
                                   negative_objects=neg_objects_str, suffix=suffix_template,
                                   pos_count=pos_count, fruit_word=fruit_word)
        answer = pos_count
    
    elif setting == "object_classes":
        # Randomly choose number of fruits and non-fruits (1 to 3 each).
        fruit_count = random.randint(1, 3)
        nonfruit_count = random.randint(1, 3)
        fruit_objs = random.sample(fruits, fruit_count)
        nonfruit_objs = random.sample(non_fruits, nonfruit_count)
        fruit_objects_str = format_object_list(fruit_objs)
        nonfruit_objects_str = format_object_list(nonfruit_objs)
        fruit_word = "fruit" if fruit_count == 1 else "fruits"
        template = random.choice(templates["object_classes"])
        sentence = template.format(prefix=prefix, fruit_objects=fruit_objects_str,
                                   nonfruit_objects=nonfruit_objects_str, suffix=suffix_template,
                                   fruit_count=fruit_count, fruit_word=fruit_word)
        answer = fruit_count
    
    elif setting == "multiple_people":
        # Randomly choose number of fruits for myself and for the other person.
        my_count = random.randint(1, 3)
        other_count = random.randint(1, 5)
        my_objs = random.sample(fruits, my_count)
        other_objs = random.sample(fruits, other_count)
        my_objects_str = format_object_list(my_objs)
        other_objects_str = format_object_list(other_objs)
        fruit_word = "fruit" if other_count == 1 else "fruits"
        name = random.choice(names)
        template = random.choice(templates["multiple_people"])
        sentence = template.format(prefix=prefix, my_objects=my_objects_str,
                                   name=name, other_objects=other_objects_str,
                                   count=other_count, fruit_word=fruit_word)
        answer = other_count
    else:
        sentence = "Invalid setting."
        answer = None
    
    # Return both the generated sentence and the correct answer.
    return {"sentence": sentence, "answer": answer}

def get_number_logit_stats(logits, correct_answer_tokens, candidate_token_ids):
    """
    Computes evaluation statistics for the counting task.
    
    Args:
      logits (torch.Tensor): model logits of shape [batch, seq_length, vocab_size] 
                             (or [batch, vocab_size]). The logits at the final prompt token are used.
      correct_answer_tokens (torch.Tensor): tensor of shape [batch] with the token IDs corresponding 
                             to the correct answer.
      candidate_token_ids (list or torch.Tensor): token IDs corresponding to the candidate numbers.
    
    Returns:
      stats (dict): Dictionary with "mean_margin", "median_margin", "min_margin", and "accuracy".
    """
    if logits.ndim == 3:
        # Use the logits for the next token prediction.
        logits = logits[:, -1, :]
    
    batch_size = logits.size(0)
    correct_logits = logits.gather(1, correct_answer_tokens.unsqueeze(1))  # [batch, 1]
    
    candidate_logits = logits[:, candidate_token_ids]  # [batch, 5]
    candidate_ids_tensor = torch.tensor(candidate_token_ids, device=logits.device)
    mask = candidate_ids_tensor.unsqueeze(0).expand(batch_size, -1) == correct_answer_tokens.unsqueeze(1)
    
    # Mask the correct candidate so we can find the highest competing logit.
    candidate_logits_masked = candidate_logits.masked_fill(mask, -1e9)
    competitor_max_logits, _ = candidate_logits_masked.max(dim=1, keepdim=True)
    
    margins = (correct_logits - competitor_max_logits).squeeze(1)  # [batch]
    mean_margin = margins.mean()
    median_margin = margins.median()
    min_margin = margins.min()
    
    # Accuracy: does the correct token have the highest logit among candidates?
    predicted_indices = candidate_logits.argmax(dim=1)  # indices 0-4.
    predicted_token_ids = candidate_ids_tensor[predicted_indices]
    accuracy = (predicted_token_ids == correct_answer_tokens).float().mean()
    
    return {
        "mean_margin": mean_margin.item(),
        "median_margin": median_margin.item(),
        "min_margin": min_margin.item(),
        "accuracy": accuracy.item()
    }

# ---------------------------
# Load model and set parameters.
# ---------------------------
model = HookedTransformer.from_pretrained("gpt2-small").cuda()
model.set_use_attn_result(True)

# ---------------------------
# Obtain candidate token IDs for "1" to "5".
# (Note: GPT-2 tokenization usually requires a leading space.)
candidate_numbers = ["1", "2", "3", "4", "5"]   
candidate_token_ids = [
    model.tokenizer.encode(" " + num, add_special_tokens=False)[0] for num in candidate_numbers
]
print("Candidate token IDs:", candidate_token_ids)

# ---------------------------
# Generate a dataset of examples.
# ---------------------------
N = 100  # You can set this to a larger number as needed.
dataset = [generate_counting_examples(seed=i, setting="default") for i in range(N)]
prompts = [example["sentence"] for example in dataset]
answers = [example["answer"] for example in dataset]

# Convert the numeric answer (e.g., 3) to the corresponding token ID.
correct_answer_token_ids = [candidate_token_ids[ans - 1] for ans in answers]
# Make sure the tensor is on the same device as the model.
correct_answer_tokens = torch.tensor(correct_answer_token_ids).cuda()

# ---------------------------
# Tokenize the prompts.
# ---------------------------
# First tokenize the full prompts
full_tokens = model.to_tokens(prompts).cuda()

# Remove the last two tokens (period and number) to create truncated prompts
# This leaves the prompt in a state where the model should naturally predict the number next
tokens = full_tokens[:, :-2]

# ---------------------------
# Evaluation before fine-tuning.
# ---------------------------
batch_size = 4
all_final_logits = []
with torch.no_grad():
    for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Evaluating batches", unit="batch"):
        batch_tokens = tokens[i:i+batch_size]
        logits = model(batch_tokens)
        # Use the logits from the final token in each sequence.
        final_logits = logits[:, batch_tokens.shape[1]-1, :]
        all_final_logits.append(final_logits)
all_final_logits = torch.cat(all_final_logits, dim=0)

logit_stats = get_number_logit_stats(all_final_logits, correct_answer_tokens, candidate_token_ids)
print("Before fine-tuning:")
print(f"Logit stats: {logit_stats}")

# ---------------------------
# Fine-tuning section.
# ---------------------------
# We fine tune the model for a few epochs on the counting task.
model.train()  # Set the model to training mode.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

num_epochs = 3  # Fine-tune for 3 epochs (adjust as needed)
for epoch in tqdm(range(num_epochs), desc="Training epochs", unit="epoch"):
    total_loss = 0
    model.train()
    for i in tqdm(range(0, tokens.shape[0], batch_size), desc=f"Epoch {epoch+1} batches", leave=False, unit="batch"):
        optimizer.zero_grad()
        batch_tokens = tokens[i:i+batch_size]
        batch_targets = correct_answer_tokens[i:i+batch_size]
        logits = model(batch_tokens)
        # Predict the next token using the logits from the final position.
        final_logits = logits[:, batch_tokens.shape[1]-1, :]
        loss = criterion(final_logits, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

# ---------------------------
# Evaluation after fine-tuning.
# ---------------------------
model.eval()
all_final_logits = []
with torch.no_grad():
    for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Evaluating batches", unit="batch"):
        batch_tokens = tokens[i:i+batch_size]
        logits = model(batch_tokens)
        final_logits = logits[:, batch_tokens.shape[1]-1, :]
        all_final_logits.append(final_logits)
all_final_logits = torch.cat(all_final_logits, dim=0)

logit_stats = get_number_logit_stats(all_final_logits, correct_answer_tokens, candidate_token_ids)
print("After fine-tuning:")
print(f"Logit stats: {logit_stats}")