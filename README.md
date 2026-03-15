# EvoCompute-Function-Optimizer

Evolutionary Algorithm (EA) implementation in Python for **maximizing** two benchmark functions and comparing 6 parent/survival selection combinations.

## Objective Functions

1. **Function 1 (Sphere)**  
   \(f(x, y) = x^2 + y^2\), with bounds: \(-5 \le x \le 5\), \(-5 \le y \le 5\)

2. **Function 2 (Rosenbrock-like)**  
   \(f(x, y) = 100(x^2 - y)^2 + (1 - x)^2\), with bounds: \(-2 \le x \le 2\), \(-1 \le y \le 3\)

Expected maximization behavior:
- Function 1 naturally approaches maximum fitness near **50**.
- Function 2 naturally approaches maximum fitness near **2500+** within bounds.

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

For each function and combination, the script runs 10 runs and exports run-level tables for selected generations:

`1, 5, 10, 15, 20, 25, 30, 35, 40`

---

## Project Structure

- `ea_assignment.py` — full EA experiment runner + CSV export + plotting
- `generate_plots.py` — legacy plotting utility (optional)
- `results/` — run-level CSV outputs
- `plots/` — generated PNG plots (individual + summary)

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

For each function (`function_1_sphere`, `function_2_rosenbrock`) and each combination:

- `results/<function_name>/<combo_name>_Average_Avg_Fit.csv`
- `results/<function_name>/<combo_name>_Average_Best_Fit.csv`

where `<combo_name>` is one of:

- `fps__truncation`
- `rbs__truncation`
- `binary_tournament__truncation`
- `fps__binary_tournament`
- `rbs__binary_tournament`
- `binary_tournament__binary_tournament`

The script also generates high-resolution PNG files in:

- `plots/function_1_sphere/`
- `plots/function_2_rosenbrock/`

including individual-combination plots and two summary plots per function:
- `summary_averages_of_all_combinations.png`
- `summary_best_fit_of_all_combinations.png`

## CSV Schema

Each exported run-level CSV has the schema:

- `Generation`
- `Run 1`
- `Run 2`
- `Run 3`
- `Run 4`
- `Run 5`
- `Run 6`
- `Run 7`
- `Run 8`
- `Run 9`
- `Run 10`
- `Average` (row-wise mean of Run 1..Run 10)

## Notes

- Core EA logic is implemented from scratch.
- EA core uses only `numpy`, `pandas`, and `random`.
- Plotting uses `matplotlib` + `seaborn`.
