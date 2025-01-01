import argparse
import sys
import torch
import warnings
import deepspeed
from transformers import AutoTokenizer
from tqdm import tqdm
from eval.perplexity import compute_perplexity
from eval.eval_passkey import load_model
from dcis import load_data, modify_factors, init_factors


def main(args):
    models = [x[0] for x in args.model]

    tokenizer = AutoTokenizer.from_pretrained(
        models[0], model_max_length=sys.maxsize, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    input_texts = load_data(args, tokenizer)

    if args.tokens_step:
        tokens = [x for x in range(
            args.min_tokens, args.max_tokens + 1, args.tokens_step)]
    else:
        tokens = [args.min_tokens]
        while args.min_tokens < args.max_tokens:
            point = tokens[-1] * 2
            if point <= args.max_tokens:
                tokens.append(point)
            else:
                break

    results = []
    for model in tqdm(models, desc="Model", leave=False, disable=args.hide_progress):
        torch.cuda.empty_cache()

        loaded = load_model(model, args)

        if args.model_name not in ["llama", "mistral"] and not args.modified:
            factors = torch.load(args.factors)
            print("factors:\n", factors)
            loaded = modify_factors(loaded, args.model_name, args.factor, factors)

        result = []
        for max_length in tokens:
            ppl = compute_perplexity(
                model=loaded, tokenizer=tokenizer, encodings=input_texts, 
                add_start_token=tokenizer.bos_token is not None, max_length=max_length
            )['mean_perplexity']
            print(f"{model}: {max_length}={ppl}")
            result.append(ppl)

        result.insert(0, model)
        results.append(result)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f",{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str, default="text")
    parser.add_argument("--paraquet", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--tokens-step", type=int)
    parser.add_argument("--model-name", type=str, default="llama")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--hide-progress", action="store_true")
    parser.add_argument("--factor", type=float)
    parser.add_argument("--factors", type=str)
    parser.add_argument("--sliding-window-attention", type=int)
    parser.add_argument("--modified", action="store_true")
    parser.add_argument("--local_rank", type=int)
    main(parser.parse_args())
