"""Evolutionary Algorithm (EA) script (MAXIMIZATION version).

What this script does:
1) Maximizes two benchmark functions.
2) Runs all 6 required parent/survival selection combinations.
3) Executes 10 independent runs per combination for 40 generations.
4) Exports run-level CSV tables at generations: 1, 5, 10, 15, 20, 25, 30, 35, 40.
5) Produces presentation-ready plots:
    - Individual-combination plots (2 lines each: average avg-fit, average best-fit)
    - Function-level summary plots for all 6 combinations.
"""

from pathlib import Path
import random
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =========================
# Assignment Configuration
# =========================
POP_SIZE = 10
OFFSPRING_SIZE = 10
GENERATIONS = 40
RUNS = 10

MUTATION_STEP = 0.25
MUTATION_RATE_PER_GENE = 0.20  # probabilistic mutation; not always applied

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = BASE_DIR / "plots"

LOG_GENERATIONS = [1, 5, 10, 15, 20, 25, 30, 35, 40]


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
    """Return objective values (higher is better, i.e., maximization)."""
    return np.array([objective_fn(ind) for ind in population], dtype=float)


def fitness_to_selection_weights(fitness_values: np.ndarray) -> List[float]:
    """
    Convert maximization fitness values to positive roulette-wheel weights.
    Higher weight => higher probability of being selected.
    """
    min_val = float(np.min(fitness_values))
    shifted = fitness_values - min_val + 1e-12
    return shifted.tolist()


def select_one_fps(population: Sequence[np.ndarray], fitness_values: np.ndarray) -> np.ndarray:
    """Fitness Proportionate Selection (roulette wheel)."""
    weights = fitness_to_selection_weights(fitness_values)
    idx = random.choices(range(len(population)), weights=weights, k=1)[0]
    return population[idx]


def select_one_rbs(population: Sequence[np.ndarray], fitness_values: np.ndarray) -> np.ndarray:
    """Rank-Based Selection (best rank gets largest probability)."""
    n = len(population)

    # Rank by fitness descending (best first)
    sorted_indices = np.argsort(-fitness_values)

    # Best gets weight n, worst gets 1
    rank_weights = np.zeros(n, dtype=float)
    for rank_position, idx in enumerate(sorted_indices):
        rank_weights[idx] = n - rank_position

    chosen_idx = random.choices(range(n), weights=rank_weights.tolist(), k=1)[0]
    return population[chosen_idx]


def select_one_binary_tournament(population: Sequence[np.ndarray], fitness_values: np.ndarray) -> np.ndarray:
    """Binary Tournament Selection."""
    i, j = random.sample(range(len(population)), 2)
    winner = i if fitness_values[i] >= fitness_values[j] else j
    return population[winner]


def select_parent(population: Sequence[np.ndarray], fitness_values: np.ndarray, scheme: str) -> np.ndarray:
    """Dispatch parent selection scheme."""
    if scheme == "fps":
        return select_one_fps(population, fitness_values)
    if scheme == "rbs":
        return select_one_rbs(population, fitness_values)
    if scheme == "binary_tournament":
        return select_one_binary_tournament(population, fitness_values)
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
    fitness_values: np.ndarray,
    parent_selection_scheme: str,
    bounds: np.ndarray,
    offspring_size: int,
) -> List[np.ndarray]:
    """Generate offspring via parent selection + crossover + mutation."""
    offspring = []
    for _ in range(offspring_size):
        p1 = select_parent(population, fitness_values, parent_selection_scheme)
        p2 = select_parent(population, fitness_values, parent_selection_scheme)
        child = crossover(p1, p2)
        child = mutate(child, bounds)
        offspring.append(child)
    return offspring


def survival_truncation(
    combined_population: Sequence[np.ndarray],
    combined_fitness_values: np.ndarray,
    survivors: int,
) -> List[np.ndarray]:
    """Truncation for maximization: keep highest-fitness individuals."""
    best_indices = np.argsort(-combined_fitness_values)[:survivors]
    return [combined_population[idx] for idx in best_indices]


def survival_binary_tournament(
    combined_population: Sequence[np.ndarray],
    combined_fitness_values: np.ndarray,
    survivors: int,
) -> List[np.ndarray]:
    """
    Binary tournament survival without replacement from the candidate pool.
    """
    candidate_indices = list(range(len(combined_population)))
    next_population = []

    while len(next_population) < survivors:
        i, j = random.sample(candidate_indices, 2)
        winner = i if combined_fitness_values[i] >= combined_fitness_values[j] else j
        next_population.append(combined_population[winner])
        candidate_indices.remove(winner)

    return next_population


def apply_survival_selection(
    combined_population: Sequence[np.ndarray],
    combined_fitness_values: np.ndarray,
    scheme: str,
    survivors: int,
) -> List[np.ndarray]:
    """Dispatch survival selection scheme."""
    if scheme == "truncation":
        return survival_truncation(combined_population, combined_fitness_values, survivors)
    if scheme == "binary_tournament":
        return survival_binary_tournament(combined_population, combined_fitness_values, survivors)
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

    Returns (for maximization):
        best_so_far_per_generation: shape (generations,)
        avg_current_population_per_generation: shape (generations,)
    """
    population = initialize_population(pop_size, bounds)

    best_so_far_values: List[float] = []
    avg_current_values: List[float] = []

    best_so_far = -float("inf")

    for _ in range(generations):
        parent_fitness = evaluate_population(population, objective_fn)

        offspring = generate_offspring(
            population=population,
            fitness_values=parent_fitness,
            parent_selection_scheme=parent_selection_scheme,
            bounds=bounds,
            offspring_size=offspring_size,
        )

        combined_population = list(population) + offspring
        combined_fitness = evaluate_population(combined_population, objective_fn)

        # Best-so-far in maximization
        gen_best = float(np.max(combined_fitness))
        best_so_far = max(best_so_far, gen_best)

        # Survival selection to next generation (exactly pop_size survivors)
        population = apply_survival_selection(
            combined_population=combined_population,
            combined_fitness_values=combined_fitness,
            scheme=survival_selection_scheme,
            survivors=pop_size,
        )

        # Average fitness of current population after survival
        current_fitness = evaluate_population(population, objective_fn)
        acp = float(np.mean(current_fitness))

        best_so_far_values.append(best_so_far)
        avg_current_values.append(acp)

    return np.array(best_so_far_values, dtype=float), np.array(avg_current_values, dtype=float)


# =========================
# Experiment Export + Plotting
# =========================
def combo_label(parent_sel: str, survival_sel: str) -> str:
    return f"{parent_sel}__{survival_sel}"


def save_run_level_metric_csv(
    out_path: Path,
    metric_matrix: np.ndarray,
    selected_generations: Sequence[int],
) -> None:
    """Save one run-level table with schema: Generation | Run 1..Run 10 | Average."""
    selected_idx = [gen - 1 for gen in selected_generations]
    selected_values = metric_matrix[:, selected_idx].T  # shape: (len(gens), RUNS)

    data = {"Generation": list(selected_generations)}
    for run_idx in range(RUNS):
        data[f"Run {run_idx + 1}"] = selected_values[:, run_idx]

    df = pd.DataFrame(data)
    run_cols = [f"Run {i}" for i in range(1, RUNS + 1)]
    df["Average"] = df[run_cols].mean(axis=1)

    df = df.round(8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def plot_individual_combination(
    function_name: str,
    combo_name: str,
    avg_avg_fit: np.ndarray,
    avg_best_fit: np.ndarray,
) -> None:
    """Plot one combination with exactly two lines: average avg-fit and average best-fit."""
    sns.set_theme(style="whitegrid", context="talk")
    generations = np.arange(1, GENERATIONS + 1)

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.plot(generations, avg_avg_fit, label="Average Avg. Fit", linewidth=2.6, linestyle="--")
    ax.plot(generations, avg_best_fit, label="Average Best Fit", linewidth=2.8, linestyle="-")

    title_fn = function_name.replace("_", " ").title()
    title_combo = combo_name.replace("__", " + ").replace("_", " ").title()
    ax.set_title(f"{title_fn} | {title_combo}", fontsize=16, pad=12)
    ax.set_xlabel("Generations", fontsize=13)
    ax.set_ylabel("Fitness values", fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(title="Metric", fontsize=11, title_fontsize=11, loc="best", frameon=True)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    out_dir = PLOTS_DIR / function_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{combo_name}_individual_metrics.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_all_combinations(
    function_name: str,
    combo_to_series: Dict[str, np.ndarray],
    metric_title: str,
    out_filename: str,
) -> None:
    """Plot all 6 combinations together for one metric."""
    sns.set_theme(style="whitegrid", context="talk")
    generations = np.arange(1, GENERATIONS + 1)
    palette = sns.color_palette("tab10", n_colors=len(SELECTION_COMBINATIONS))
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (parent_sel, survival_sel) in enumerate(SELECTION_COMBINATIONS):
        combo_name = combo_label(parent_sel, survival_sel)
        series = combo_to_series[combo_name]
        display_label = combo_name.replace("__", " + ").replace("_", " ").title()

        ax.plot(
            generations,
            series,
            label=display_label,
            color=palette[idx],
            linestyle=linestyles[idx],
            linewidth=2.4,
        )

    title_fn = function_name.replace("_", " ").title()
    ax.set_title(f"{title_fn} | {metric_title}", fontsize=16, pad=12)
    ax.set_xlabel("Generations", fontsize=13)
    ax.set_ylabel("Fitness values", fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(title="Combinations", fontsize=10, title_fontsize=11, loc="best", frameon=True)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    out_dir = PLOTS_DIR / function_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / out_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_all_experiments() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for function_name, cfg in FUNCTIONS.items():
        objective_fn = cfg["fn"]
        bounds = cfg["bounds"]

        function_results_dir = RESULTS_DIR / function_name
        function_results_dir.mkdir(parents=True, exist_ok=True)

        summary_avg_by_combo: Dict[str, np.ndarray] = {}
        summary_best_by_combo: Dict[str, np.ndarray] = {}

        for parent_sel, survival_sel in SELECTION_COMBINATIONS:
            combo_name = combo_label(parent_sel, survival_sel)

            # Store full 40-generation histories for each run
            best_fit_runs: List[np.ndarray] = []
            avg_fit_runs: List[np.ndarray] = []

            for _ in range(RUNS):
                best_history, avg_history = run_single_ea(
                    objective_fn=objective_fn,
                    bounds=bounds,
                    parent_selection_scheme=parent_sel,
                    survival_selection_scheme=survival_sel,
                    pop_size=POP_SIZE,
                    offspring_size=OFFSPRING_SIZE,
                    generations=GENERATIONS,
                )
                best_fit_runs.append(best_history)
                avg_fit_runs.append(avg_history)

            best_fit_matrix = np.vstack(best_fit_runs)  # (RUNS, GENERATIONS)
            avg_fit_matrix = np.vstack(avg_fit_runs)    # (RUNS, GENERATIONS)

            # Means across runs for plotting
            avg_best_fit = np.mean(best_fit_matrix, axis=0)
            avg_avg_fit = np.mean(avg_fit_matrix, axis=0)
            summary_best_by_combo[combo_name] = avg_best_fit
            summary_avg_by_combo[combo_name] = avg_avg_fit

            # Required run-level CSV exports (selected generations only)
            avg_fit_csv = function_results_dir / f"{combo_name}_Average_Avg_Fit.csv"
            best_fit_csv = function_results_dir / f"{combo_name}_Average_Best_Fit.csv"

            save_run_level_metric_csv(avg_fit_csv, avg_fit_matrix, LOG_GENERATIONS)
            save_run_level_metric_csv(best_fit_csv, best_fit_matrix, LOG_GENERATIONS)

            # Required individual combination plot
            plot_individual_combination(function_name, combo_name, avg_avg_fit, avg_best_fit)

            print(f"Saved: {avg_fit_csv}")
            print(f"Saved: {best_fit_csv}")

        # Required function-level summary plots
        plot_summary_all_combinations(
            function_name=function_name,
            combo_to_series=summary_avg_by_combo,
            metric_title="Averages of all combinations",
            out_filename="summary_averages_of_all_combinations.png",
        )
        plot_summary_all_combinations(
            function_name=function_name,
            combo_to_series=summary_best_by_combo,
            metric_title="Best fit of all combinations",
            out_filename="summary_best_fit_of_all_combinations.png",
        )

        print(f"Saved summary plots for: {function_name}")


if __name__ == "__main__":
    run_all_experiments()
