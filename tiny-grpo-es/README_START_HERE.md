# START HERE: ES vs PPO for Sparse Reward RL

## 🎯 Project Goal

Compare **Evolution Strategies (ES)** parameter-space optimization against **PPO** action-space RL in environments with sparse, outcome-only rewards and long-horizon credit assignment.

## ⚡ Quick Start (5 minutes)

### 1. Install Dependencies

You already have NumPy and PyTorch! Just need visualization:

```bash
cd tiny-grpo-es
pip install matplotlib seaborn
```

### 2. Verify Everything Works

```bash
python quick_test.py
```

This runs a mini ES experiment in ~2 minutes to verify installation.

### 3. Run Your First Real Experiment

```bash
# Quick test (15-20 minutes)
python compare_methods.py --env simple --trials 3 --iterations 50

# Full proof-of-life (1-2 hours)
python compare_methods.py --env simple --trials 5 --iterations 100
```

This compares:
- **Random** (baseline)
- **ES** (parameter-space optimization)
- **PPO** (action-space RL)

Results are saved to `./results/` with plots and statistics.

## 📊 What You're Testing

### Environment: GridWorld
- **State**: Agent position on N×N grid
- **Actions**: {up, down, left, right}
- **Goal**: Navigate from bottom-left to top-right
- **Obstacles**: Random blockers
- **Reward**: +1.0 only at goal (sparse!)

### Methods

**Evolution Strategies (ES)**
- Perturb policy weights → evaluate → estimate gradient → update
- No backpropagation needed
- Naturally handles sparse rewards

**PPO (Proximal Policy Optimization)**
- Standard policy gradient with value function
- Action-space optimization
- Baseline for comparison

**Random**
- Uniform random actions
- Lower bound on performance

## 📈 Expected Results

| Method | Success Rate | Variance |
|--------|--------------|----------|
| Random | 0-5% | Low |
| ES | 30-50% | Low |
| PPO | 20-40% | Higher |

**Key insight**: ES should show more stable learning (lower variance) than PPO in sparse reward settings.

## 📁 File Guide

| File | What It Does |
|------|--------------|
| `quick_test.py` | ✅ Verify installation (run this first!) |
| `compare_methods.py` | 🔬 Main experiment (compare all methods) |
| `train_es_gridworld.py` | 🧬 Train ES only |
| `train_ppo_gridworld.py` | 🎯 Train PPO only |
| `gridworld_env.py` | 🌍 Environment definitions |
| `policy_network.py` | 🧠 Neural networks |
| **`PROJECT_SUMMARY.md`** | 📖 Detailed project overview |
| **`SETUP_INSTRUCTIONS.md`** | 📚 Detailed usage guide |
| `README_GRIDWORLD.md` | 🔬 Full documentation |

## 🎓 For Your Week 1 Deliverable

### Experiment Command
```bash
python compare_methods.py \
    --env simple \
    --size 8 \
    --obstacles 8 \
    --trials 5 \
    --iterations 100 \
    --output ./results/week1
```

### What You Get
1. **Plots**: `./results/week1/comparison_GridWorld.png`
   - Box plots comparing rewards, success rates, steps
2. **Statistics**: Console output with mean ± std
3. **Learning Curves**: Track reward vs iterations

### Success Criteria (From Your Proposal)
- ✅ ES achieves stable reward improvement
- ✅ Outperforms random baseline significantly
- ✅ Comparable or better than PPO
- ✅ Lower variance across trials

## 🔧 Customize Your Experiments

### Change Environment Difficulty
```bash
# Easier (faster to train)
python compare_methods.py --size 5 --obstacles 3 --iterations 50

# Harder (longer horizon)
python compare_methods.py --size 10 --obstacles 15 --iterations 200

# Key-door environment (even harder!)
python compare_methods.py --env harder --iterations 200
```

### Change Algorithm Parameters

Edit `train_es_gridworld.py` or `train_ppo_gridworld.py`:

**ES**:
- `N`: Population size (more = stable but slower)
- `sigma`: Noise scale (smaller = finer search)
- `alpha`: Learning rate

**PPO**:
- `lr_policy`: Learning rate
- `gamma`: Discount factor
- `n_steps`: Rollout length

## 🐛 Troubleshooting

### "No module named X"
```bash
pip install matplotlib seaborn
```

### Training too slow
- Use smaller grid: `--size 5`
- Fewer trials: `--trials 3`
- Fewer iterations: `--iterations 50`

### Not learning
- Increase iterations: `--iterations 200`
- Adjust hyperparameters (see SETUP_INSTRUCTIONS.md)

## 📚 Learn More

1. **Quick overview**: Read `PROJECT_SUMMARY.md` (5 min)
2. **Detailed setup**: Read `SETUP_INSTRUCTIONS.md` (10 min)
3. **Full documentation**: Read `README_GRIDWORLD.md` (20 min)

## 🚀 Next Steps After Week 1

1. ✅ Analyze variance: Is ES more stable?
2. ✅ Visualize policies: Where do agents get stuck?
3. ✅ Test harder environment: Key-door mechanic
4. ✅ Tune hyperparameters: Optimize performance
5. ✅ Try larger networks: Scalability test

## 💡 Key Insights

### Why ES Works for Sparse Rewards

Traditional RL (like PPO) struggles with sparse rewards because:
- Policy gradients have high variance
- Credit assignment is difficult
- Action-space exploration is inefficient

ES handles this better because:
- **Parameter space**: Directly optimizes weights, not actions
- **Zeroth-order**: Only needs function evaluations (forward passes)
- **Population-based**: Averages over many perturbations (variance reduction)
- **Global search**: Can escape local optima more easily

### Your Technical Approach

From your proposal:
> "We plan to use Evolution Strategies–style parameter perturbations to estimate zeroth-order gradients over policy parameters"

✅ **Implemented!** The code does exactly this:
1. Sample perturbations: θ' = θ + σε
2. Evaluate: R(θ')
3. Estimate gradient: ∇J ≈ (1/Nσ) Σ R(θ')ε
4. Update: θ ← θ + α∇J

## 📞 Need Help?

1. Check `SETUP_INSTRUCTIONS.md` for detailed usage
2. Check `PROJECT_SUMMARY.md` for project overview
3. Look at code comments in `.py` files
4. Check the original repo: [tiny-grpo-es](https://github.com/damek/tiny-grpo-es)

## ✨ You're All Set!

Everything is ready for your proof-of-life experiment. Just run:

```bash
python quick_test.py                    # Verify (2 min)
python compare_methods.py --trials 5   # Experiment (1-2 hrs)
```

Good luck! 🎓🚀
