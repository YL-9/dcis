import gc
from tqdm import tqdm
import torch


def compute_perplexity(
    encodings, model, tokenizer, add_start_token: bool = True, device=None, max_length=None, sliding_window=256, truncate=True, aggressive_memory=False, hide_progress=False,
):
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595"""
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if add_start_token:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts), disable=hide_progress)
    nlls = []
    for encoding_index in range(0, len(encoded_texts)):

        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window):

            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)

            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
            
            if aggressive_memory:
                outputs = None
                input_ids = None
                target_ids = None
                gc.collect()
                torch.cuda.empty_cache()

            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}
