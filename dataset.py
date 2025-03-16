from itertools import product
import random

def generate_dataset_per_permutation(samples_per_perm=100):
    """
    Generate a dataset with exactly 'samples_per_perm' samples for each unique permutation
    of conditions:
      - is_plural: Boolean (False=Singular, True=Plural)
      - is_negated: Boolean (False=Affirmative, True=Negated)
      - has_prefix: Boolean (False=No prefix, True=With time prefix)
      - is_pronoun: Boolean (False=Name, True=Pronoun)
      - tense: "present" or "past"
      - use_irregular: Boolean (False=Regular verb, True=Irregular verb)
      
    Returns a list of example dictionaries.
    """
    dataset = []
    
    # Extended list of names and pronouns.
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
                              
    # For each permutation, generate the fixed number of samples.
    for is_plural, is_negated, has_prefix, is_pronoun, tense, use_irregular in conditions:
        for _ in range(samples_per_perm):
            # Select subject.
            if is_pronoun:
                subject = random.choice(plural_pronouns) if is_plural else random.choice(singular_pronouns)
            else:
                subject = random.choice(plural_names) if is_plural else random.choice(singular_names)
            
            # Choose a prefix based on 'has_prefix' and 'tense'.
            if has_prefix:
                if tense == "past":
                    prefix = random.choice(past_prefixes)
                else:  # present tense
                    prefix = random.choice(present_prefixes)
            else:
                prefix = ""
            
            # Verb selection.
            if use_irregular:
                verb_key = random.choice(list(irregular_verbs.keys()))
                forms = irregular_verbs[verb_key]
                if tense == "present":
                    if is_negated:
                        # Negated present: auxiliary forces base form.
                        correct_verb = forms["present_plural"]
                        incorrect_verb = forms["present_singular"]
                    else:
                        if is_plural:
                            correct_verb = forms["present_plural"]
                            incorrect_verb = forms["present_singular"]
                        else:
                            correct_verb = forms["present_singular"]
                            incorrect_verb = forms["present_plural"]
                else:  # past tense
                    if is_negated:
                        # Negated past: auxiliary "did not" forces base form.
                        correct_verb = forms["present_plural"]
                        incorrect_verb = forms["past"]
                    else:
                        correct_verb = forms["past"]
                        incorrect_verb = forms["present_plural"]
            else:
                # Use regular verbs.
                if tense == "present":
                    if is_negated:
                        correct_verb = random.choice(reg_present_plural)
                        incorrect_verb = random.choice(reg_present_singular)
                    else:
                        if is_plural:
                            correct_verb = random.choice(reg_present_plural)
                            incorrect_verb = random.choice(reg_present_singular)
                        else:
                            correct_verb = random.choice(reg_present_singular)
                            incorrect_verb = random.choice(reg_present_plural)
                else:  # past tense
                    if is_negated:
                        correct_verb = random.choice(reg_present_plural)
                        incorrect_verb = random.choice(reg_past)
                    else:
                        correct_verb = random.choice(reg_past)
                        incorrect_verb = random.choice(reg_present_plural)
            
            # Construct prompt.
            if is_negated:
                if tense == "past":
                    aux = "did not "
                else:
                    aux = "does not " if not is_plural else "do not "
                prompt = f"{prefix}{subject} {aux}"
            else:
                prompt = f"{prefix}{subject} "
            
            dataset.append({
                "prompt": prompt,
                "correct_verb": correct_verb,
                "incorrect_verb": incorrect_verb,
                "is_negated": is_negated,
                "is_plural": is_plural,
                "subject": subject,
                "prefix": prefix.strip(),
                "is_pronoun": is_pronoun,
                "tense": tense,
                "use_irregular": use_irregular
            })
            
    return dataset