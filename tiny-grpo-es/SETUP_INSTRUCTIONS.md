# Setup Instructions for GridWorld ES Experiments

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
cd tiny-grpo-es

# Option A: Use pip (recommended)
pip install torch numpy matplotlib seaborn

# Option B: Full requirements (includes wandb for logging)
pip install -r requirements_gridworld.txt
```

**Note**: You don't need the LLM dependencies (transformers, accelerate) for the gridworld experiments. Those are only for the original GRPO code.

### 2. Verify Installation

Run the quick test to make sure everything works:

```bash
python quick_test.py
```

This should complete in ~2 minutes and show:
- ✓ Environment works
- ✓ Policy network works  
- ✓ ES training works

### 3. Run Your First Experiment

**Option A: Quick comparison (15-20 minutes)**
```bash
python compare_methods.py --env simple --trials 3 --iterations 50
```

**Option B: Full proof-of-life experiment (~1-2 hours)**
```bash
python compare_methods.py --env simple --trials 5 --iterations 100
```

This will:
1. Train ES policy (parameter-space optimization)
2. Train PPO policy (action-space RL baseline)
3. Compare both to random baseline
4. Generate plots showing results

## Understanding the Output

### During Training

You'll see output like:
```
Iter 20/100: reward_mean=0.234, eval_reward=0.412, eval_success=0.40, eval_steps=28.3
```

Where:
- `reward_mean`: Average reward across perturbations (ES) or episodes (PPO)
- `eval_reward`: Evaluation reward on separate test episodes
- `eval_success`: Fraction of episodes that reached the goal
- `eval_steps`: Average steps to complete episode

### Final Results

After all trials complete:
```
FINAL RESULTS (mean ± std)
RANDOM  : reward=-0.234±0.045, success=0.020±0.014, steps=50.0±0.0
ES      : reward=0.412±0.102, success=0.450±0.085, steps=32.1±4.2
PPO     : reward=0.338±0.156, success=0.380±0.120, steps=35.8±6.1
```

A plot will be saved to `./results/comparison_GridWorld.png` showing:
1. Reward comparison (higher is better)
2. Success rate (higher is better)
3. Steps to goal (lower is better)

## Experiment Configurations

### Proof of Life (Week 1)

**Goal**: Demonstrate ES can learn in sparse reward setting

```bash
# Simple gridworld: sparse reward at goal only
python compare_methods.py \
    --env simple \
    --size 8 \
    --obstacles 8 \
    --trials 5 \
    --iterations 100
```

**Expected results**:
- Random: ~0-5% success
- ES: ~30-50% success (stable across trials)
- PPO: ~20-40% success (higher variance)

### Harder Environment (Long Horizon)

**Goal**: Test credit assignment over longer horizons

```bash
# Key-door gridworld: must collect key then reach goal
python compare_methods.py \
    --env harder \
    --size 10 \
    --obstacles 15 \
    --trials 5 \
    --iterations 200
```

**Expected results**:
- Random: ~0% success (extremely unlikely to solve by chance)
- ES: ~10-30% success
- PPO: ~5-20% success (struggles with long horizon)

## Individual Method Training

### Train ES Only

```bash
python train_es_gridworld.py
```

Edit hyperparameters in the `main()` function:
- `N`: Population size (more = more stable but slower)
- `sigma`: Noise scale (lower = finer search)
- `alpha`: Learning rate (higher = faster but less stable)
- `T`: Iterations (more = better performance)

### Train PPO Only

```bash
python train_ppo_gridworld.py
```

Edit hyperparameters in the `main()` function:
- `n_steps`: Rollout length
- `lr_policy`: Policy learning rate
- `gamma`: Discount factor (lower = myopic, higher = farsighted)
- `n_iterations`: Training iterations

## Tracking with Weights & Biases (Optional)

If you want to track experiments with wandb:

```bash
# Login to wandb (creates free account if needed)
wandb login

# Run with wandb enabled (it's on by default in individual training scripts)
python train_es_gridworld.py
```

To disable wandb, edit the training scripts and set `log_wandb=False`.

## Troubleshooting

### "No module named 'wandb'"

Either:
1. Install it: `pip install wandb`
2. Or disable it: In training scripts, set `log_wandb=False`

### Training is slow

- Reduce `N` (population size for ES) - try N=10
- Reduce `n_steps` (rollout length for PPO) - try n_steps=64
- Use smaller grid: `--size 5 --obstacles 3`
- Use GPU if available (automatically detected)

### ES not improving

- Increase iterations (`--iterations 200`)
- Tune `sigma` (try 0.02 or 0.1)
- Tune `alpha` (try 0.005 or 0.02)
- Check if environment is too hard (reduce obstacles)

### PPO not improving

- Increase `n_steps` for longer rollouts
- Tune learning rate (try `lr_policy=1e-3` or `1e-4`)
- Adjust entropy coefficient for more exploration
- Check reward shaping (sparse rewards are hard for PPO)

## Next Steps

After getting basic results:

1. **Analyze learning curves**: Look at reward vs iteration plots
2. **Compare variance**: ES should have lower variance across trials
3. **Test harder environments**: Increase size, obstacles, or use key-door
4. **Visualize policies**: Modify code to render learned policies
5. **Extend to other environments**: Tower defense, symbolic reasoning, etc.

## File Overview

| File | Purpose |
|------|---------|
| `gridworld_env.py` | Environment definitions (simple & harder) |
| `policy_network.py` | Neural network policies |
| `train_es_gridworld.py` | ES training loop |
| `train_ppo_gridworld.py` | PPO training loop |
| `compare_methods.py` | Main experiment script |
| `quick_test.py` | Installation verification |
| `README_GRIDWORLD.md` | Project documentation |

## Questions?

See `README_GRIDWORLD.md` for more detailed documentation on:
- Algorithm descriptions
- Hyperparameter tuning
- Environment details
- Expected results
- Future directions
