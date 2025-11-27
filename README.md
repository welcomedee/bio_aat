# Neuro-Swarm vs. Backpropagation

## Project Overview

This project demonstrates how **Particle Swarm Optimization (PSO)**, a nature-inspired algorithm, can achieve competitive or superior accuracy compared to standard backpropagation by **directly optimizing the accuracy metric**.

### The Key Advantage

**Standard Backpropagation** requires differentiable loss functions (like Cross-Entropy) to calculate gradients. This means it **cannot directly optimize accuracy** because accuracy is not differentiable.

**Particle Swarm Optimization** does not rely on gradients. It can optimize **any metric directly**, including accuracy. This flexibility makes it superior for certain optimization tasks.

## Dataset

**Wisconsin Breast Cancer Diagnostic Dataset**
- Binary classification (Malignant vs. Benign)
- 30 features
- 569 samples
- Clear class separation

## Models Compared

### 1. Baseline: Standard Backpropagation
- Architecture: Input(30) → Hidden(10) → Output(2)
- Optimizer: Adam
- Loss Function: Cross-Entropy (proxy for accuracy)
- Training: Standard gradient descent

### 2. Optimized: Particle Swarm Optimization
- Architecture: Input(30) → Hidden(10) → Output(2) *(same as baseline)*
- Optimizer: PSO with 50 particles
- Objective Function: **Accuracy (direct optimization)**
- Training: Swarm-based evolution over 300 iterations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Run the Comparison

```bash
python neuro_swarm_comparison.py
```

This will:
- Load and preprocess the Wisconsin Breast Cancer dataset
- Train the baseline model (backpropagation)
- Train the PSO-optimized model
- Display detailed comparison metrics
- Save results to `results.npy`

### Step 2: Generate Visualizations

```bash
python visualize_results.py
```

This will generate:
- `comparison_results.png`: Comprehensive comparison dashboard
- `pso_convergence.png`: Detailed PSO convergence analysis

## Expected Results

The PSO-optimized model should achieve:
- **Competitive or better accuracy** than backpropagation
- **Direct optimization** of the accuracy metric
- **Proof of concept** that nature-inspired algorithms can match or exceed traditional methods

## Key Insights

### Why PSO Can Be Better

1. **Direct Metric Optimization**: PSO can optimize accuracy directly, while backpropagation must use proxy loss functions
2. **No Gradient Requirements**: Works with non-differentiable activation functions
3. **Global Search**: Swarm exploration helps escape local minima
4. **Flexibility**: Can optimize any custom metric without mathematical constraints

### Trade-offs

- **Training Time**: PSO typically takes longer than backpropagation
- **Scalability**: Better suited for smaller networks (due to high-dimensional search space)
- **Convergence**: May require tuning of PSO hyperparameters (c1, c2, w)

## Project Structure

```
bio_aat/
├── neuro_swarm_comparison.py   # Main comparison script
├── visualize_results.py        # Visualization generator
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── results.npy                 # Saved results (generated)
├── comparison_results.png      # Comparison dashboard (generated)
└── pso_convergence.png         # PSO convergence plot (generated)
```

## Customization

### Adjust PSO Parameters

In `neuro_swarm_comparison.py`, modify:

```python
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=n_weights, options=options)
cost, best_pos = optimizer.optimize(f_global, iters=300, verbose=True)
```

- `c1`: Cognitive parameter (particle's own experience)
- `c2`: Social parameter (swarm's collective knowledge)
- `w`: Inertia weight (exploration vs. exploitation)
- `n_particles`: Number of particles in the swarm
- `iters`: Number of optimization iterations

### Change Network Architecture

Modify the architecture in both models:

```python
# For baseline
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# For PSO (update n_hidden)
n_hidden = 10
```

## References

- **Dataset**: [UCI Machine Learning Repository - Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **PySwarms**: [PySwarms Documentation](https://pyswarms.readthedocs.io/)
- **Paper**: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.

## License

This project is for educational purposes.
