# EvoCompute-Function-Optimizer

Evolutionary Algorithm (EA) implementation in Python for minimizing two benchmark functions and comparing selection scheme combinations.

## Objective Functions

1. **Function 1 (Sphere)**  
   \(f(x, y) = x^2 + y^2\), with bounds: \(-5 \le x \le 5\), \(-5 \le y \le 5\)

2. **Function 2 (Rosenbrock-like)**  
   \(f(x, y) = 100(x^2 - y)^2 + (1 - x)^2\), with bounds: \(-2 \le x \le 2\), \(-1 \le y \le 3\)

## EA Configuration

- Population size: **10**
- Offspring per generation: **10**
- Survival pool: **20** (parents + offspring)
- Survivors each generation: **10**
- Generations per run: **40**
- Runs per combination: **10**
- Mutation: **probabilistic** \(\pm 0.25\) per gene

## Selection Combinations Tested

1. FPS + Truncation
2. RBS + Truncation
3. Binary Tournament + Truncation
4. FPS + Binary Tournament
5. RBS + Binary Tournament
6. Binary Tournament + Binary Tournament

## Tracked Metrics

- **Best-so-far (BSF)** per generation
- **Average-of-current-population (ACP)** per generation

For each function and combination, the script averages BSF and ACP across 10 runs and exports CSV data for all 40 generations.

---

## Project Structure

- `ea_assignment.py` — EA implementation and experiment runner
- `generate_plots.py` — plotting utility for presentation-ready figures
- `results/` — aggregated CSV outputs
- `plots/` — generated PNG plots

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn
```

## Run Experiments

```bash
python ea_assignment.py
```

This generates:

- `results/function_1_sphere_aggregated_results.csv`
- `results/function_2_rosenbrock_aggregated_results.csv`

## Generate Plots

```bash
python generate_plots.py
```

This generates high-resolution (300 DPI) PNG files in `plots/`:

- `function_1_sphere_average_bsf.png`
- `function_1_sphere_average_avg_fit.png`
- `function_2_rosenbrock_average_bsf.png`
- `function_2_rosenbrock_average_avg_fit.png`

## CSV Schema

Each row corresponds to one generation of one selection combination.

Columns:

- `Function`
- `Parent_Selection`
- `Survival_Selection`
- `Combination`
- `Generation`
- `Average_BSF`
- `Average_Avg_Fit`

## Notes

- Core EA logic is implemented from scratch.
- Only `numpy`, `pandas`, `random` are used for the EA core.
- Plotting uses `matplotlib` + `seaborn`.
