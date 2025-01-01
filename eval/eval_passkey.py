import argparse
import random
import re
import sys
import torch
import warnings
from transformers import AutoTokenizer, pipeline, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
from dcis import modify_factors, init_factors

# from https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py


def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 10000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def test_model(pipe, prompt_text, pass_key):
    response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=10)[
        0]["generated_text"][len(prompt_text):]
    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pass_key = int(re.search(r'\d+', response).group())
    except:
        pass_key = response[:20]

    return pass_key


def load_model(model_path, args):
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
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            config=config,
            use_flash_attention_2=True,
        )
        return model
    config = config_cls.from_pretrained(model_path)
    factors = torch.load(args.factors)
    print("factors:\n", factors)
    config.rope_scaling = {
        "type": "dcis",
        "factor": args.factor,
        "factors": factors
    }
    if args.sliding_window_attention:
        config.sliding_window = args.sliding_window_attention
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        config=config,
        use_flash_attention_2=True,
    )
    return model


def main(args):
    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained(
        models[0], model_max_length=sys.maxsize, padding_side="right", trust_remote_code=True)

    if args.fixed_length:
        lengths = [args.fixed_length]
        tokens = [len(tokenizer.encode(generate_prompt(args.fixed_length)[0]))]
        print(f"Prompt is {tokens[0]} tokens")
    else:
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

        lengths = []
        last_n = 0
        for target in tqdm(tokens, desc="Determining sequence lengths"):
            num_tokens = 0
            n = last_n
            while num_tokens < target:
                last_n = n
                n += args.length_step
                prompt = generate_prompt(n)[0]
                num_tokens = len(tokenizer.encode(prompt))
            lengths.append(last_n)

    results = []
    for model in tqdm(models, desc="Model", leave=False):
        torch.cuda.empty_cache()

        loaded = load_model(model, args)

        if args.model_name not in ["llama", "mistral"] and not args.modified:
            factors = torch.load(args.factors)
            print("factors:\n", factors)
            loaded = modify_factors(loaded, args.model_name, args.factor, factors)

        pipe = pipeline("text-generation", model=loaded,
                        tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)

        result = [0] * len(lengths)
        for i, length in tenumerate(lengths, desc="Lengths", leave=False):
            torch.cuda.empty_cache()
            for _ in trange(0, args.iterations, desc="Iterations", leave=False):
                torch.cuda.empty_cache()
                prompt_text, pass_key = generate_prompt(length)
                num_tokens = len(pipe.tokenizer.encode(prompt_text))
                answer = test_model(pipe, prompt_text, pass_key)
                if answer == pass_key:
                    result[i] += 1
            result[i] /= args.iterations
            print(f"{model}: {tokens[i]}={int(result[i]*100)}%")

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
    parser.add_argument("--model-name", type=str, default="llama")
    parser.add_argument("--fixed-length", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--tokens-step", type=int)
    parser.add_argument("--length-step", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--factor", type=float)
    parser.add_argument("--factors", type=str)
    parser.add_argument("--sliding-window-attention", type=int)
    parser.add_argument("--modified", action="store_true")
    main(parser.parse_args())
