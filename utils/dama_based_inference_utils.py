#Based on causal_trace and gender_trace

import torch


def dama_inference(model, tokenizer, prompts):
    # prompts must be an iterable of strings
    inp = dama_make_inputs(tokenizer, prompts)
    #print(f"inputs :\n{inp}")
    with torch.no_grad():
        base_score = dama_pronoun_probs(model, tokenizer, inp)
    #print(f"base_score.size() : {base_score.size()}")
    #print(f"base_score : {base_score}")
    return base_score


def dama_make_inputs(tokenizer, prompts, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def dama_pronoun_probs(model, tokenizer, inp):
    out = model(**inp)
    probs = dama_get_pronoun_probabilities(out.logits, tokenizer)
    return probs


def dama_get_pronoun_probabilities(output, tokenizer):
    probabilities = torch.softmax(output[:, -1, :], dim=1)

    pronoun_tokens = ('he', 'she')
    pronouns_indices = torch.tensor([tokenizer.encode(pronoun) for pronoun in pronoun_tokens]).reshape((-1,))

    pron_prob = probabilities[:, pronouns_indices]

    return pron_prob
