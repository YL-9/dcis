import argparse
import copy
import json
import sys
import warnings
import datasets
import torch
import math
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from eval.perplexity import compute_perplexity


def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)

def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def yarn_factors(s, head_dim, base, original_max_position_embeddings, beta_fast, beta_slow):
    inv_freq_extrapolation = torch.ones(head_dim // 2)
    inv_freq_interpolation = inv_freq_extrapolation * s
    low, high = _yarn_find_correction_range(beta_fast, beta_slow, head_dim, base, original_max_position_embeddings)
    inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, head_dim // 2).float())
    inv_freq = s / (inv_freq_extrapolation * (1 - inv_freq_mask) + inv_freq_interpolation * inv_freq_mask)
    return inv_freq


def init_factors(scaling_factor, init_type, dim, base, original_max_position_embeddings, beta_fast, beta_slow):
    factors = None
    if init_type == 'ones':
        factors = torch.ones(dim // 2)
    elif init_type == 'yarn':
        factors = yarn_factors(scaling_factor, dim, base, original_max_position_embeddings, beta_fast, beta_slow)
    elif init_type == 'ntk':
        factors = (scaling_factor ** (torch.arange(0, dim, 2, dtype=torch.float32) / (dim - 2)))
    elif init_factors == 'pi':
        factors = torch.full([1, dim//2], scaling_factor)
    else:
        raise RuntimeError("init_type must be one of [ones, yarn, ntk, pi]")
    return factors


def load_model(args, path):
    if args.model_name == "llama":
        from modeling_source.modeling_llama import LlamaForCausalLM
        from modeling_source.configuration_llama import LlamaConfig
        config_cls = LlamaConfig
        model_cls = LlamaForCausalLM
    elif args.model_name == "mistral":
        from modeling_source.modeling_mistral import MistralForCausalLM
        from modeling_source.configuration_mistral import MistralConfig
        config_cls = MistralConfig
        model_cls = MistralForCausalLM
    else:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            config=config,
            use_flash_attention_2=True,
        )
        return model
    config = config_cls.from_pretrained(args.model)
    if args.sliding_window_attention:
        config.sliding_window = args.sliding_window_attention
    model = model_cls.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        config=config,
        use_flash_attention_2=True,
    )
    return model

def load_data(args, tokenizer):
    if args.tokenized:
        try:
            input_texts = datasets.load_from_disk(args.tokenized)
        except:
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split)
    else:
        input_texts = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split)

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            return example

        input_texts = input_texts.map(tokenize)
        if args.save_tokenized:
            input_texts.save_to_disk(args.save_tokenized)
            print(f"Saved tokenized dataset to {args.save_tokenized}")
            return

    if args.dataset_min_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens)
    if args.samples:
        input_texts = input_texts[:args.samples]

    return input_texts


def output_cur_factors(factors, init_type, scaling_factor, num, model_name, cur_range):
    output_path = f'./factors/{model_name}/{scaling_factor}_{init_type}_{num}.pt'
    torch.save(factors.clone(), output_path)
    save_range = {"cur_range": cur_range, "num": num}
    output_path = f'./factors/{model_name}/{scaling_factor}_{init_type}_{num}.json'
    with open(output_path, 'w') as f:
        json.dump(save_range, f)


def modify_factors(model, model_name, scaling_factor, factors):
    if model_name == "llama":
        from modeling_source.modeling_llama import LlamaDCISScalingRotaryEmbedding
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaDCISScalingRotaryEmbedding(
                each.self_attn.head_dim,
                max_position_embeddings=each.self_attn.max_position_embeddings,
                scaling_factor=scaling_factor,
                each_dim_factors=copy.deepcopy(factors),
                base=each.self_attn.rope_theta,
            )
    elif model_name == "mistral":
        from modeling_source.modeling_mistral import MistralDCISScalingRotaryEmbedding
        for each in model.model.layers:
            each.self_attn.rotary_emb = MistralDCISScalingRotaryEmbedding(
                each.self_attn.head_dim,
                max_position_embeddings=each.self_attn.max_position_embeddings,
                scaling_factor=scaling_factor,
                each_dim_factors=copy.deepcopy(factors),
                base=each.self_attn.rope_theta,
            )
    else:
        from modeling_source.patch import LlamaDCISScalingRotaryEmbedding
        for each in model.model.layers:
            each.self_attn.rotary_emb = LlamaDCISScalingRotaryEmbedding(
                each.self_attn.head_dim,
                max_position_embeddings=each.self_attn.max_position_embeddings,
                scaling_factor=scaling_factor,
                each_dim_factors=copy.deepcopy(factors),
                base=each.self_attn.rope_theta,
            )

    return model


def dcis(model, data, tokenizer, cur_factors, cur_range, mnum, num, itercnt, max_length, scaling_factor, model_name):
    new_factors = copy.deepcopy(cur_factors)
    new_range = []
    jj = range(0, mnum, num)
    for j in reversed(jj):
        ppl = []
        step = (cur_range[j//num][1] - cur_range[j//num][0]) / (itercnt - 1)
        kk = [cur_range[j//num][0] + step * i for i in range(itercnt)]
        for k in kk:
            print('k=', k)
            tmp_factors = copy.deepcopy(new_factors)
            tmp_factors[j:j+num] += k
            tmp_factors = torch.clamp(tmp_factors, min=0.0001)

            model = modify_factors(model, model_name, scaling_factor, tmp_factors)

            tmp_ppl = compute_perplexity(
                model=model, encodings=data, tokenizer=tokenizer,
                add_start_token=tokenizer.bos_token is not None, max_length=max_length,
            )['mean_perplexity']
            print("ppl:", tmp_ppl)
            if tmp_ppl < 100:
                ppl.append([tmp_ppl, k])
        
        if len(ppl) == 0:
            new_range.insert(0, [cur_range[j//num][0], cur_range[j//num][1]])
            new_range.insert(0, [cur_range[j//num][0], cur_range[j//num][1]])
            continue

        ppl.sort(key=lambda x: x[0])
        print('d:', j, '~', j+num)
        print('ppl:', ppl)

        new_factors[j:j+num] += ppl[0][1]
        low = high = ppl[0][1]
        for k in range(min(3, len(ppl))):
            low = min(low, ppl[k][1])
            high = max(high, ppl[k][1])
        low -= step
        high += step
        new_range.insert(0, [low-ppl[0][1], high-ppl[0][1]])
        new_range.insert(0, [low-ppl[0][1], high-ppl[0][1]])
    
    return new_factors, new_range


def main(args):
    model_path = args.model
    model = load_model(args, model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
            model_path, model_max_length=sys.maxsize, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    data =  load_data(args,tokenizer)

    scaling_factor = args.factor
    init_type = args.init_type
    model_name = args.model_name
    head_dim = model.model.layers[0].self_attn.head_dim
    base = model.model.layers[0].self_attn.rope_theta
    max_length = args.original_max_position_embeddings * int(scaling_factor)
    mnum = head_dim // 2
    num = mnum // 2
    itercnt = args.itercnt
    cur_factors = init_factors(scaling_factor, init_type , head_dim, base, 
                               args.original_max_position_embeddings, 
                               args.beta_fast, args.beta_slow)
    cur_range = [[-4*1.25, 4*1.25], [-4*1.25, 4*1.25]]

    if args.longrope:
        from longrope import search_lambda_factors
        search_lambda_factors(model=model, data=data, tokenizer=tokenizer, 
                              extension_ratio=scaling_factor, max_length=max_length, 
                              model_name=model_name, head_dim=head_dim, base=base, 
                              original_max_position_embeddings=args.original_max_position_embeddings, 
                              beta_fast=args.beta_fast, beta_slow=args.beta_slow)
        return

    while num >= 1:
        output_cur_factors(cur_factors, init_type, scaling_factor, num, model_name, cur_range)
        print('num:', num)
        print('cur_factors:', cur_factors)
        print('cur_range:', cur_range)
        cur_factors, cur_range = dcis(model=model, data=data, tokenizer=tokenizer, 
                                cur_factors=cur_factors, cur_range=cur_range, mnum=mnum, 
                                num=num, itercnt=itercnt, max_length=max_length, 
                                scaling_factor=scaling_factor, model_name=model_name)
        num //= 2

    output_cur_factors(cur_factors, init_type, scaling_factor, num, model_name, cur_range)
    print('num:', num)
    print('cur_factors:', cur_factors)
    print('cur_range:', cur_range)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str)
    args.add_argument("--dataset", type=str)
    args.add_argument("--samples", type=int, default=5)
    args.add_argument("--factor", type=float, default=1)
    args.add_argument("--model-name", type=str, default="llama")
    args.add_argument("--init-type", type=str, default="yarn")
    args.add_argument("--original-max-position-embeddings", type=int, default=4096)
    args.add_argument("--itercnt", type=int, default=10)
    args.add_argument("--tokenized", type=str)
    args.add_argument("--split", type=str, default="validation")
    args.add_argument("--subset", type=str)
    args.add_argument("--feature", type=str, default="text")
    args.add_argument("--save-tokenized", type=str)
    args.add_argument("--dataset-min-tokens", type=int)
    args.add_argument("--paraquet", type=str)
    args.add_argument("--longrope", action="store_true")
    args.add_argument("--beta-fast", type=float, default=32)
    args.add_argument("--beta-slow", type=float, default=1)
    args.add_argument("--sliding-window-attention", type=int)
    main(args.parse_args())
