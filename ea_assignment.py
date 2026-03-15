"""
Evolutionary Algorithm (EA) assignment script.

Implements an EA from scratch to minimize two benchmark functions:
1) Sphere:      f(x, y) = x^2 + y^2
2) Rosenbrock:  f(x, y) = 100(x^2 - y)^2 + (1 - x)^2

For each function, it runs 6 parent/survival selection combinations,
10 independent runs per combination, and 40 generations per run.

Outputs aggregated CSV files (average over 10 runs) for:
- Best-so-far (BSF)
- Average-of-current-population (ACP)
"""

import os
import random
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# =========================
# Assignment Configuration
# =========================
POP_SIZE = 10
OFFSPRING_SIZE = 10
GENERATIONS = 40
RUNS = 10

MUTATION_STEP = 0.25
MUTATION_RATE_PER_GENE = 0.20  # probabilistic mutation; not always applied

RESULTS_DIR = "results"


# =========================
# Objective Functions
# =========================
def sphere(individual: np.ndarray) -> float:
    """Function 1: f(x, y) = x^2 + y^2"""
    x, y = individual
    return float(x**2 + y**2)


def rosenbrock_like(individual: np.ndarray) -> float:
    """Function 2: f(x, y) = 100(x^2 - y)^2 + (1 - x)^2"""
    x, y = individual
    return float(100 * (x**2 - y) ** 2 + (1 - x) ** 2)


FUNCTIONS: Dict[str, Dict[str, object]] = {
    "function_1_sphere": {
        "fn": sphere,
        "bounds": np.array([[-5.0, 5.0], [-5.0, 5.0]], dtype=float),
    },
    "function_2_rosenbrock": {
        "fn": rosenbrock_like,
        "bounds": np.array([[-2.0, 2.0], [-1.0, 3.0]], dtype=float),
    },
}


SELECTION_COMBINATIONS: List[Tuple[str, str]] = [
    ("fps", "truncation"),
    ("rbs", "truncation"),
    ("binary_tournament", "truncation"),
    ("fps", "binary_tournament"),
    ("rbs", "binary_tournament"),
    ("binary_tournament", "binary_tournament"),
]


# =========================
# EA Utilities
# =========================
def initialize_population(pop_size: int, bounds: np.ndarray) -> List[np.ndarray]:
    """Randomly initialize population within bounds."""
    population = []
    for _ in range(pop_size):
        x = random.uniform(bounds[0, 0], bounds[0, 1])
        y = random.uniform(bounds[1, 0], bounds[1, 1])
        population.append(np.array([x, y], dtype=float))
    return population


def evaluate_population(population: Sequence[np.ndarray], objective_fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """Return objective values (lower is better)."""
    return np.array([objective_fn(ind) for ind in population], dtype=float)


def objective_to_selection_weights(obj_values: np.ndarray) -> List[float]:
    """
    Convert minimization objective values to positive selection weights.
    Higher weight => higher probability of being selected.
    """
    # Safe for non-negative objectives; both assignment objectives are >= 0.
    return [1.0 / (1.0 + v) for v in obj_values]


def select_one_fps(population: Sequence[np.ndarray], obj_values: np.ndarray) -> np.ndarray:
    """Fitness Proportionate Selection (roulette wheel)."""
    weights = objective_to_selection_weights(obj_values)
    idx = random.choices(range(len(population)), weights=weights, k=1)[0]
    return population[idx]


def select_one_rbs(population: Sequence[np.ndarray], obj_values: np.ndarray) -> np.ndarray:
    """Rank-Based Selection (best rank gets largest probability)."""
    n = len(population)

    # Rank by objective ascending (best first)
    sorted_indices = np.argsort(obj_values)

    # Best gets weight n, worst gets 1
    rank_weights = np.zeros(n, dtype=float)
    for rank_position, idx in enumerate(sorted_indices):
        rank_weights[idx] = n - rank_position

    chosen_idx = random.choices(range(n), weights=rank_weights.tolist(), k=1)[0]
    return population[chosen_idx]


def select_one_binary_tournament(population: Sequence[np.ndarray], obj_values: np.ndarray) -> np.ndarray:
    """Binary Tournament Selection."""
    i, j = random.sample(range(len(population)), 2)
    winner = i if obj_values[i] <= obj_values[j] else j
    return population[winner]


def select_parent(population: Sequence[np.ndarray], obj_values: np.ndarray, scheme: str) -> np.ndarray:
    """Dispatch parent selection scheme."""
    if scheme == "fps":
        return select_one_fps(population, obj_values)
    if scheme == "rbs":
        return select_one_rbs(population, obj_values)
    if scheme == "binary_tournament":
        return select_one_binary_tournament(population, obj_values)
    raise ValueError(f"Unknown parent selection scheme: {scheme}")


def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """
    Arithmetic crossover: child = alpha*p1 + (1-alpha)*p2.
    """
    alpha = random.random()
    child = alpha * parent1 + (1.0 - alpha) * parent2
    return child.astype(float)


def mutate(individual: np.ndarray, bounds: np.ndarray, mutation_rate: float = MUTATION_RATE_PER_GENE) -> np.ndarray:
    """
    Probabilistic mutation with +/- 0.25 per gene.
    Mutation is NOT automatically applied to every gene.
    """
    mutated = individual.copy()
    for g in range(mutated.shape[0]):
        if random.random() < mutation_rate:
            mutated[g] += random.choice([-MUTATION_STEP, MUTATION_STEP])

    # Keep within search bounds
    mutated[0] = min(max(mutated[0], bounds[0, 0]), bounds[0, 1])
    mutated[1] = min(max(mutated[1], bounds[1, 0]), bounds[1, 1])
    return mutated


def generate_offspring(
    population: Sequence[np.ndarray],
    obj_values: np.ndarray,
    parent_selection_scheme: str,
    bounds: np.ndarray,
    offspring_size: int,
) -> List[np.ndarray]:
    """Generate offspring via parent selection + crossover + mutation."""
    offspring = []
    for _ in range(offspring_size):
        p1 = select_parent(population, obj_values, parent_selection_scheme)
        p2 = select_parent(population, obj_values, parent_selection_scheme)
        child = crossover(p1, p2)
        child = mutate(child, bounds)
        offspring.append(child)
    return offspring


def survival_truncation(
    combined_population: Sequence[np.ndarray],
    combined_obj_values: np.ndarray,
    survivors: int,
) -> List[np.ndarray]:
    """Truncation: keep the best objective-value individuals."""
    best_indices = np.argsort(combined_obj_values)[:survivors]
    return [combined_population[i] for i in best_indices]


def survival_binary_tournament(
    combined_population: Sequence[np.ndarray],
    combined_obj_values: np.ndarray,
    survivors: int,
) -> List[np.ndarray]:
    """
    Binary tournament survival without replacement from the candidate pool.
    """
    candidate_indices = list(range(len(combined_population)))
    next_population = []

    while len(next_population) < survivors:
        i, j = random.sample(candidate_indices, 2)
        winner = i if combined_obj_values[i] <= combined_obj_values[j] else j
        next_population.append(combined_population[winner])
        candidate_indices.remove(winner)

    return next_population


def apply_survival_selection(
    combined_population: Sequence[np.ndarray],
    combined_obj_values: np.ndarray,
    scheme: str,
    survivors: int,
) -> List[np.ndarray]:
    """Dispatch survival selection scheme."""
    if scheme == "truncation":
        return survival_truncation(combined_population, combined_obj_values, survivors)
    if scheme == "binary_tournament":
        return survival_binary_tournament(combined_population, combined_obj_values, survivors)
    raise ValueError(f"Unknown survival selection scheme: {scheme}")


# =========================
# EA Run Logic
# =========================
def run_single_ea(
    objective_fn: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    parent_selection_scheme: str,
    survival_selection_scheme: str,
    pop_size: int = POP_SIZE,
    offspring_size: int = OFFSPRING_SIZE,
    generations: int = GENERATIONS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one EA trial.

    Returns:
        bsf_per_generation: shape (generations,)
        acp_per_generation: shape (generations,)
    """
    population = initialize_population(pop_size, bounds)

    bsf_values = []
    acp_values = []

    best_so_far = float("inf")

    for _ in range(generations):
        parent_obj = evaluate_population(population, objective_fn)

        offspring = generate_offspring(
            population=population,
            obj_values=parent_obj,
            parent_selection_scheme=parent_selection_scheme,
            bounds=bounds,
            offspring_size=offspring_size,
        )

        combined_population = list(population) + offspring
        combined_obj = evaluate_population(combined_population, objective_fn)

        # Best-so-far tracks the best ever found up to this point
        gen_best = float(np.min(combined_obj))
        best_so_far = min(best_so_far, gen_best)

        # Survival selection to next generation (exactly pop_size survivors)
        population = apply_survival_selection(
            combined_population=combined_population,
            combined_obj_values=combined_obj,
            scheme=survival_selection_scheme,
            survivors=pop_size,
        )

        # ACP uses current population after survival
        current_obj = evaluate_population(population, objective_fn)
        acp = float(np.mean(current_obj))

        bsf_values.append(best_so_far)
        acp_values.append(acp)

    return np.array(bsf_values, dtype=float), np.array(acp_values, dtype=float)


# =========================
# Experiment Orchestration
# =========================
def run_all_experiments() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for fn_name, cfg in FUNCTIONS.items():
        objective_fn = cfg["fn"]
        bounds = cfg["bounds"]

        rows = []

        for parent_sel, survival_sel in SELECTION_COMBINATIONS:
            combo_label = f"{parent_sel}__{survival_sel}"

            bsf_runs = []
            acp_runs = []

            for _ in range(RUNS):
                bsf, acp = run_single_ea(
                    objective_fn=objective_fn,
                    bounds=bounds,
                    parent_selection_scheme=parent_sel,
                    survival_selection_scheme=survival_sel,
                    pop_size=POP_SIZE,
                    offspring_size=OFFSPRING_SIZE,
                    generations=GENERATIONS,
                )
                bsf_runs.append(bsf)
                acp_runs.append(acp)

            bsf_mean = np.mean(np.vstack(bsf_runs), axis=0)
            acp_mean = np.mean(np.vstack(acp_runs), axis=0)

            for gen in range(1, GENERATIONS + 1):
                rows.append(
                    {
                        "Function": fn_name,
                        "Parent_Selection": parent_sel,
                        "Survival_Selection": survival_sel,
                        "Combination": combo_label,
                        "Generation": gen,
                        "Average_BSF": bsf_mean[gen - 1],
                        "Average_Avg_Fit": acp_mean[gen - 1],
                    }
                )

        result_df = pd.DataFrame(rows)

        # Nicely formatted: sort and round for readability
        result_df = result_df.sort_values(["Combination", "Generation"]).reset_index(drop=True)
        result_df["Average_BSF"] = result_df["Average_BSF"].round(8)
        result_df["Average_Avg_Fit"] = result_df["Average_Avg_Fit"].round(8)

        out_path = os.path.join(RESULTS_DIR, f"{fn_name}_aggregated_results.csv")
        result_df.to_csv(out_path, index=False)

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_all_experiments()
