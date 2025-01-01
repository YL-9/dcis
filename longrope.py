import random
import numpy as np
import torch
import copy
from eval.perplexity import compute_perplexity
from dcis import modify_factors, init_factors


def initialize_population(population_size, extension_ratio, head_dim, base, original_max_position_embeddings, beta_fast, beta_slow):
    population = []

    yarn_factors = init_factors(extension_ratio, 'yarn', head_dim, base, original_max_position_embeddings, beta_fast, beta_slow)
    population.append(yarn_factors)

    for _ in range(population_size - 1):
        factors = mutate_indv(yarn_factors, extension_ratio)
        population.append(factors)

    return population


def evaluate_individual(model, data, individual):
    model.lambda_factors = individual
    perplexities = []

    for seq in data:
        input_ids = seq.unsqueeze(0)
        output = model(input_ids)
        perplexity = torch.exp(torch.mean(output))
        perplexities.append(perplexity.item())

    return np.mean(perplexities)


def evaluate_population(model, data, tokenizer, population, max_length, extension_ratio, model_name):
    perplexities = []
    for individual in population:
        model = modify_factors(model, model_name, extension_ratio, individual)
        perplexity = compute_perplexity(
                model=model, encodings=data, tokenizer=tokenizer,
                add_start_token=tokenizer.bos_token is not None, max_length=max_length,
            )['mean_perplexity']
        perplexities.append(perplexity)
    return perplexities


def select_topk(population, perplexities, cnt, k):
    indices = np.argsort(perplexities)[:k]

    print(f"cnt: {cnt}\n", population[indices[0]])
    output_path = f'./factors/longrope/{cnt}.pt'
    torch.save(population[indices[0]].clone(), output_path)

    return [population[i] for i in indices]

def mutate_indv(parent, extension_ratio):
    list_step = 0.01
    evo_list = np.arange(1.0, 1.0 + extension_ratio + list_step, list_step)
    new_factors = copy.deepcopy(parent)
    flag = True

    while flag or not np.all(np.diff(new_factors) >= 0):
        flag = False
        for dim in range(new_factors.shape[0]):
            if np.random.rand() < 0.3:
                if dim == 0:
                    evo_list_curr = np.arange(1.0, new_factors[dim + 1], list_step)
                elif dim == new_factors.shape[0] - 1:
                    evo_list_curr = np.arange(new_factors[dim - 1], evo_list.max() + list_step, list_step)
                else:
                    evo_list_curr = np.arange(new_factors[dim - 1], new_factors[dim + 1] + list_step, list_step)

                if evo_list_curr.shape[0] > 0:
                    layer_index = np.random.randint(0, evo_list_curr.shape[0])
                    new_factors = copy.deepcopy(new_factors)
                    new_factors[dim] = evo_list_curr[layer_index]
    
    return new_factors

def mutate(parents, num_mutations, head_dim, extension_ratio):
    mutated_population = []
    for _ in range(num_mutations):
        mutated_population.append(mutate_indv(parents[_], extension_ratio))
    return mutated_population


def crossover(parents, num_crossovers, head_dim):
    crossover_population = []
    for __ in range(num_crossovers):
        parent1, parent2 = random.sample(parents, 2)
        new_factors = copy.deepcopy(parent1)
        for _ in range(20):
            for i in range(new_factors.shape[0]):
                if np.random.rand() < 0.3:
                    new_factors = copy.deepcopy(new_factors)
                    if np.random.rand() < 0.5:
                        new_factors[i] = parent2[i]
                    if not np.all(np.diff(new_factors) >= 0):
                        continue
                    break

        crossover_population.append(new_factors)
    return crossover_population


def search_lambda_factors(
    model,
    data,
    tokenizer,
    extension_ratio,
    max_length,
    model_name,
    head_dim,
    base, original_max_position_embeddings, beta_fast, beta_slow, 
    population_size = 64,
    num_mutations = 16,
    num_crossovers = 16,
    max_iterations = 40,
):
    population = initialize_population(population_size, extension_ratio, head_dim, base, original_max_position_embeddings, beta_fast, beta_slow)

    for i in range(max_iterations):
        perplexities = evaluate_population(model, data, tokenizer, population, max_length, extension_ratio, model_name)
        parents = select_topk(population, perplexities, i, k=population_size // 2)
        population = parents + mutate(parents, num_mutations, head_dim, extension_ratio) + crossover(parents, num_crossovers, head_dim)
