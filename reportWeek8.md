# Evolution Strategies with Low-Rank Adaptation for Transfer Learning

## Problem Statement

**What are we optimizing?** We are training neural network policies using Evolution Strategies (ES) enhanced with Low-Rank Adaptation (LoRA) to enable efficient transfer learning in reinforcement learning environments. The core innovation is comparing **full parameter ES** against **LoRA-constrained ES** when adapting to perturbed environments. The policy network (2-layer MLP with 64 hidden units, 8,580 parameters) maps GridWorld states to action probabilities, with LoRA reducing the search space to just 324 trainable parameters—a 26.48× compression.

**Why does this problem matter?** While ES has proven effective as a gradient-free alternative to policy gradient methods, its sample efficiency and scalability remain challenging. LoRA offers a principled way to reduce ES gradient variance by constraining parameter updates to a low-rank subspace. This is especially relevant for:
1. **Transfer learning**: Adapting pretrained policies to new but related environments
2. **Curriculum learning**: Progressively training on harder task variants
3. **Sample efficiency**: Reducing the number of perturbations needed per ES iteration

**How will we measure success?** We focus on **adaptation speed** rather than just final performance. Primary metrics:
- **Time-to-threshold**: Iterations to reach 60%, 80%, 90% success rates
- **Sample efficiency**: Total environment interactions to reach performance thresholds
- **AUC (area under curve)**: Success rate vs iteration during adaptation
- **Final performance**: Success rate, average reward, and steps after adaptation

**Constraints and known risks.** LoRA's rank-1 parameterization assumes the optimal update lies in a low-dimensional subspace, which may not hold for drastic environment changes. The curriculum learning approach requires careful tuning of perturbation levels—too small and the task is trivial, too large and LoRA cannot adapt effectively.

**Data requirements.** Our experimental protocol requires:
- **GridWorld specification**: 8×8 grid, 8 randomly placed obstacles, one goal, max 50 steps per episode
- **Perturbation protocol**: 4 levels (0%, 25%, 50%, 75% of movable cells shifted by ≤1 grid cell)
- **Seeds**: 3 random seeds (11, 33, 55) for reproducibility
- **Training budget**: 100 iterations pretraining + 80 iterations adaptation = 180 iterations total
- **Evaluation frequency**: Every 10 iterations during adaptation
- **Total experiments**: 3 seeds × 4 perturbation levels × 2 methods = 24 adaptation runs

## Technical Approach

**Environment perturbation and curriculum.** We use controlled GridWorld layout perturbations where a fraction of obstacle/goal cells are shifted by up to 1 grid cell. Perturbation levels: 0% (baseline), 25%, 50%, 75% of movable cells. This creates a curriculum from near-identical to substantially different environments.

**Low-Rank Adaptation (LoRA) formulation.** For each linear layer with weights W ∈ ℝ^(d_out × d_in), LoRA adds:

```
W' = W + α · B · A
```

where A ∈ ℝ^(k × d_in) and B ∈ ℝ^(d_out × k) with rank k=1. The base weights W are frozen after pretraining, and ES optimizes only the low-rank factors (A, B), reducing gradient dimensionality from 8,580 to 324 parameters.

**Adaptation protocol.** For each random seed and perturbation level:
1. **Pretrain**: Run full-parameter ES on baseline GridWorld (100 iterations)
2. **Perturb**: Create modified environment by moving grid cells
3. **Adapt**: Continue training for 80 iterations using:
   - **Full ES** (`param_mode='all'`): Update all 8,580 parameters
   - **LoRA ES** (`param_mode='lora'`): Update only 324 LoRA parameters

Both methods start from the same pretrained checkpoint to ensure fair comparison.

**ES hyperparameters.** Pretraining and adaptation use:
- Population size N=50 (more stable than Week 4's 30-50 range)
- Noise scale σ=0.1 (balanced exploration)
- Learning rate α=0.05 (aggressive but stable with fitness normalization)
- Episodes per evaluation: 5 (reduces variance)
- Max steps per episode: 50

**Validation strategy.** We evaluate on **sparse reward** environments (no shaping) despite training with distance-based reward shaping (+0.2 for moving closer, -0.01 step penalty). This tests whether learned policies generalize to the true sparse objective.

**PyTorch implementation strategy.**
- **Parameter flattening**: All network parameters (or LoRA-only) are concatenated into a single 1D tensor for ES perturbations
- **LoRA wrapper**: Custom `LoRALinearWrapper` class wraps `nn.Linear` layers, adding low-rank factors A and B
- **Frozen base weights**: After pretraining, base layer weights are marked with `requires_grad=False` for LoRA mode
- **Checkpoint management**: Pretrained policies saved as `.pt` files, loaded into both Full ES and LoRA ES for fair comparison
- **Evaluation**: Separate sparse-reward GridWorld instances used for unbiased policy evaluation

**Code organization.**
- `src/gridworld.py`: Base `GridWorld`, `PolicyNetwork`, `LoRALinearWrapper`
- `src/es_gridworld.py`: `train_es` with `param_mode` support, `freeze_base_weights()`
- `notebooks/week7_implementation.ipynb`: Full experimental pipeline with curriculum

## Results

**Pretraining performance (100 iterations, baseline GridWorld).**
All seeds achieved near-perfect performance on the baseline environment:

| Metric | Value |
|---|---|
| Success rate | ~95-100% |
| Training time | ~2-3 minutes (CPU) |
| Pretrained checkpoint | Used for both Full ES and LoRA ES adaptation |

**Adaptation experiments (80 iterations, perturbed environments).**

Results averaged over 3 random seeds (11, 33, 55) for each perturbation level:

| Perturbation | Method | Time to 80% | Interactions to 80% | Final Success | AUC |
|---|---|---|---|---|---|
| 0% (baseline) | Full ES | 12.3 iter | 30,750 | 98.7% | 0.91 |
| 0% | **LoRA ES** | **8.7 iter** | **21,750** | 97.3% | 0.89 |
| 25% | Full ES | 18.7 iter | 46,750 | 94.0% | 0.85 |
| 25% | LoRA ES | 22.3 iter | 55,750 | 89.3% | 0.80 |
| 50% | Full ES | 28.3 iter | 70,750 | 87.3% | 0.76 |
| 50% | LoRA ES | Did not reach 80% threshold | - | 74.0% | 0.68 |
| 75% | Full ES | 42.0 iter | 105,000 | 82.7% | 0.71 |
| 75% | LoRA ES | Did not reach 80% threshold | - | 68.0% | 0.61 |

**Key findings:**

1. **LoRA dominates at low perturbation (0%)**: 29% faster to reach 80% success with similar final performance. The low-rank constraint acts as regularization.

2. **Full ES becomes necessary at high perturbation (≥50%)**: LoRA cannot reach 80% threshold within budget. The rank-1 subspace is too restrictive for large distribution shifts.

3. **Crossover at moderate perturbation (25%)**: Full ES slightly outperforms but at higher sample cost. This is the "sweet spot" where curriculum design matters most.

4. **Sample efficiency trade-off**: LoRA evaluates 26× fewer parameters per iteration but requires full-rank updates for drastic changes.

**Comparison to Week 4 (pure GridWorld) and Week 6 (Wordle):**

| Task | Method | Key Result | Sample Budget |
|---|---|---|---|
| Week 4: GridWorld | ES vs PPO | Both reach 100% success | ES: 25,000 episodes |
| Week 6: Wordle | ES vs PPO | ES 10-15%, PPO 10% best | ES: 120,000 episodes |
| **Week 8: Transfer** | **Full ES vs LoRA ES** | **LoRA 29% faster (low Δ)** | **~100,000 episodes** |

The Wordle task from Week 6 proved significantly harder (10-15% success) compared to GridWorld's near-perfect performance, highlighting the difficulty of sparse discrete action spaces with 156-word vocabularies.

**Resource usage measurements.**

| Metric | Full ES | LoRA ES |
|---|---|---|
| Training time per iteration | ~2.5 seconds | ~2.3 seconds |
| Memory usage (peak) | ~180 MB | ~160 MB |
| Episodes per second | ~40 | ~43 |
| Total compute (180 iterations) | ~7.5 minutes | ~6.9 minutes |
| Parameter perturbations per iteration | 50 × 8,580 = 429,000 | 50 × 324 = 16,200 |

LoRA's memory savings are modest (11%) since the base network is still evaluated, but the **26× reduction in perturbed parameters** is the key advantage. All experiments ran on CPU (no GPU required).

**Unexpected challenges encountered.**

1. **LoRA failure threshold**: We initially expected LoRA to degrade gracefully at higher perturbations, but it hit a sharp cliff at 50%—failing to reach 80% success at all. This suggests rank-1 is fundamentally insufficient for large distribution shifts.

2. **25% perturbation instability**: The moderate perturbation level showed high variance across seeds. Some runs favored LoRA, others Full ES, making it difficult to draw firm conclusions without more trials.

3. **Pretraining sensitivity**: Early experiments with weaker pretraining (60-70% success) led to complete adaptation failure for both methods. High-quality pretraining (>95%) appears critical for transfer learning.

4. **Curriculum tuning difficulty**: Finding the "right" perturbation levels required manual experimentation. Levels too close together (e.g., 0%, 10%, 20%) made trends unclear; too far apart (0%, 50%, 100%) missed the crossover point.

## Next Steps

**Immediate improvements (Week 9):**

1. **Adaptive rank scheduling**: Start with rank-1 LoRA for small perturbations, increase rank for larger shifts. Test k=1,2,4,8 curriculum.

2. **Multi-stage curriculum**: Chain perturbation levels (0% → 25% → 50%) with LoRA reinitialization at each stage, inheriting previous stage's adapted weights.

3. **Apply to Wordle**: The 61,468-parameter Wordle policy could benefit from LoRA's ~20× compression. Test curriculum over:
   - Vocabulary size (50 → 100 → 156 words)
   - Feedback complexity (greens only → greens+yellows → full feedback)

**Technical enhancements:**

- **Mirrored sampling** (+ε/-ε perturbations) for LoRA variance reduction
- **Adaptive σ scheduling**: Start high for exploration, decay for exploitation
- **Multi-seed robustness**: Expand from 3 to 10+ seeds for statistical significance

**Alternative approaches:**

- **Natural ES**: Use the natural gradient for LoRA updates (may reduce iteration count)
- **CMA-ES with LoRA**: Covariance matrix adaptation over low-rank subspace
- **Hybrid full+LoRA**: Alternate between full-rank and LoRA updates during adaptation

**Questions we need help with:**

1. **Rank selection theory**: Is there a principled way to choose LoRA rank based on expected distribution shift? Currently we guess k=1,2,4 empirically.

2. **Curriculum design**: How to systematically design perturbation schedules? Should we use task metrics (policy divergence) or hand-tuned levels?

3. **Scaling to Wordle**: The 61K-parameter Wordle policy has very different structure (discrete vocab). How should LoRA be applied—per layer uniformly, or with layer-specific ranks?

4. **Variance reduction**: LoRA's gradient estimates are still noisy (N=50 perturbations). Should we increase N, use mirrored sampling, or switch to Natural ES?

5. **Theoretical guarantees**: Under what conditions is rank-k LoRA sufficient for adaptation? Can we bound the approximation error?

**What we learned:**

1. **LoRA is not a silver bullet**: It excels at small distribution shifts but fails at large ones. The 26× parameter compression comes with representational capacity trade-offs.

2. **Curriculum design is critical**: The 25% perturbation level is where careful tuning matters most—neither method dominates.

3. **Transfer learning works**: Pretraining accelerates adaptation dramatically compared to training from scratch (4-5× faster in pilot experiments).

4. **ES + LoRA is complementary**: ES provides gradient-free optimization, LoRA provides low-rank structure. Together they enable sample-efficient transfer.

5. **Wordle remains challenging**: Despite vocabulary masking fixes and state embedding improvements from Week 6, the 10-15% success rate suggests we need richer state representations or LLM-based policies.

## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.

3. Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum Learning. *ICML*.

4. Mania, H., Guy, A., & Recht, B. (2018). Simple random search provides a competitive approach to reinforcement learning. *NeurIPS*.

5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
