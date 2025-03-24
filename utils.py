import torch

def get_logits(prompt, tokenizer, model):
    """Return model logits for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

def get_logit_diff(prompt: str, correct_word: str, incorrect_word: str, tokenizer, model):
    """
    Compute the logit difference (correct minus incorrect) for the next-token prediction.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :].cpu() 
    
    correct_id = tokenizer.encode(correct_word, add_prefix_space=True)[0]
    incorrect_id = tokenizer.encode(incorrect_word, add_prefix_space=True)[0]
    
    logit_correct = logits[correct_id].item()
    logit_incorrect = logits[incorrect_id].item()
    
    logit_diff = logit_correct - logit_incorrect
    return logit_diff, logit_correct, logit_incorrect
