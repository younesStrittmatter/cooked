# DTDE Grid Search System

This system provides a comprehensive grid search framework for hyperparameter optimization and reward/penalty configuration exploration for the DTDE (Dynamic Task Distribution Engine) reinforcement learning training.

## 🎯 Overview

The grid search system systematically explores different combinations of:

- **PPO Hyperparameters**: Learning rates, discount factors (γ), GAE lambda (λ), entropy coefficients, clip parameters, and value function coefficients
- **Reward Configurations**: Rewards for cutting, making salads, and delivering orders
- **Penalty Configurations**: Penalties for inefficient actions, useless actions, and destructive behaviors
- **Multiple Random Seeds**: For statistical significance and reproducibility

## 📁 Files Structure

```
├── training-DTDE-gridsearch.py      # Modified training script accepting hyperparameters as CLI args
├── analyze_grid_search_results      # Analysis launched after the grid search
├── launch_grid_search.sh            # Advanced grid search with configuration file
├── grid_search_config.sh            # Configuration file for parameter ranges
└── README_GridSearch.md             # This documentation
```

## 🚀 Quick Start

### Grid Search (Configurable)

```bash
# 1. Edit configuration file to customize parameter ranges
nano grid_search_config.sh

# 2. Run advanced grid search
./launch_grid_search.sh
```

## ⚙️ Configuration

### Basic Parameters (grid_search_config.sh)

```bash
# Base Configuration
CLUSTER="cuenca"                    # Cluster environment
MAP_NR="baseline_division_of_labor_v2"  # Game map
NUM_EPOCHS=300                      # Training epochs per experiment
NUM_AGENTS=1                        # Number of agents (1 or 2)

# Parameter Ranges
LR_VALUES=(0.0001 0.0003 0.001)    # Learning rates to test
SEEDS=(0 42 123)                   # Random seeds for reproducibility
GAMMA_VALUES=(0.95 0.99 0.995)     # Discount factors
ENT_COEF_VALUES=(0.01 0.05 0.1)    # Exploration vs exploitation

# Training Batch Parameters
TRAIN_BATCH_SIZE_VALUES=(2000 4000 8000)      # Total training batch sizes
SGD_MINIBATCH_SIZE_VALUES=(250 500 1000)      # SGD minibatch sizes  
NUM_SGD_ITER_VALUES=(5 10 20)                 # SGD iterations per batch

# Neural Network Architectures
MLP_ARCHITECTURES=(
    "small:256:128:64"      # Lightweight architecture
    "medium:512:256:128"    # Balanced architecture
    "large:512:512:256"     # High-capacity architecture
)
```

### Penalty Configurations

Format: `"name:busy_penalty:useless_action_penalty:destructive_action_penalty"`

```bash
PENALTY_CONFIGS=(
    "low:0.005:1.0:5.0"      # Lenient penalties
    "medium:0.01:2.0:10.0"   # Balanced penalties  
    "high:0.02:4.0:20.0"     # Strict penalties
)
```

### Reward Configurations

Format: `"name:cut_reward:salad_reward:deliver_reward"`

```bash
REWARD_CONFIGS=(
    "sparse:0.0:0.0:10.0"      # Only delivery rewards
    "shaped:2.0:5.0:10.0"      # Intermediate rewards
    "dense:5.0:7.0:10.0"       # Dense reward shaping
    "equal:5.0:5.0:5.0"        # Equal reward distribution
)
```

## 🔍 Hyperparameter Details

### PPO Hyperparameters

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| **GAMMA** | Discount factor for future rewards | 0.9-0.999 | Higher = more long-term focused |
| **GAE_LAMBDA** | Bias-variance tradeoff in advantage estimation | 0.9-0.99 | Higher = less bias, more variance |
| **ENT_COEF** | Entropy coefficient for exploration | 0.01-0.1 | Higher = more exploration |
| **CLIP_EPS** | PPO clip parameter for policy stability | 0.1-0.3 | Higher = allows bigger policy changes |
| **VF_COEF** | Value function loss weight | 0.25-1.0 | Balance between policy and value learning |

### Training Batch Hyperparameters

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|---------|
| **TRAIN_BATCH_SIZE** | Total samples per training iteration | 2000-8000 | Higher = more stable gradients, slower training |
| **SGD_MINIBATCH_SIZE** | Samples per SGD update | 250-1000 | Higher = more stable updates, less frequent updates |
| **NUM_SGD_ITER** | SGD iterations per training batch | 5-20 | Higher = more thorough optimization, longer training |

### Neural Network Architecture

| Architecture | Hidden Layers | Use Case | Memory Usage |
|-------------|---------------|----------|--------------|
| **Small** | [256, 128, 64] | Fast experiments, limited resources | Low |
| **Medium** | [512, 256, 128] | Balanced performance/speed | Medium |
| **Large** | [512, 512, 256] | High performance requirements | High |
| **Deep** | [256, 256, 256] | Complex state spaces | Medium |
| **Wide** | [1024, 512, 256] | Very complex environments | Very High |

### Reward Design Philosophy

- **Sparse Rewards**: Only reward final objectives (delivery)
  - *Pros*: Clear objective, no reward hacking
  - *Cons*: Harder to learn, slower convergence

- **Shaped Rewards**: Intermediate rewards for sub-goals
  - *Pros*: Faster learning, guided behavior
  - *Cons*: Risk of reward hacking, suboptimal policies

- **Dense Rewards**: Frequent feedback for all meaningful actions
  - *Pros*: Fast learning, detailed feedback
  - *Cons*: Complex reward engineering, potential for exploitation

### Penalty Strategy

- **Low Penalties**: Encourage exploration and experimentation
- **Medium Penalties**: Balanced efficiency and exploration
- **High Penalties**: Enforce strict efficiency and optimal behavior

## 📊 Results Organization

The grid search creates organized results in `grid_search_logs/`:

```
grid_search_logs/
└── gridsearch_YYYYMMDD_HHMMSS/
    ├── grid_search_summary.txt       # Overall experiment summary
    ├── output_exp1_lr0.0003_seed0_medium_shaped.txt  # Individual experiment logs
    ├── progress_exp1_lr0.0003_seed0_medium_shaped.txt # Progress tracking
    └── grid_search_results.csv       # Parsed results (after analysis)
```

## 🧪 Experiment Types

### 1. Hyperparameter Sensitivity Analysis
Test how sensitive the algorithm is to different PPO parameters:

```bash
# Focus on entropy coefficient for exploration study
ENT_COEF_VALUES=(0.001 0.01 0.05 0.1 0.2)
# Keep other parameters fixed
GAMMA_VALUES=(0.99)
GAE_LAMBDA_VALUES=(0.95)
```

### 2. Reward Shaping Study
Compare different reward strategies:

```bash
REWARD_CONFIGS=(
    "sparse:0.0:0.0:10.0"
    "light_shaping:1.0:2.0:10.0" 
    "medium_shaping:3.0:5.0:10.0"
    "heavy_shaping:8.0:8.0:10.0"
)
```

### 3. Penalty Impact Analysis
Study how penalties affect learning:

```bash
PENALTY_CONFIGS=(
    "no_penalty:0.0:0.0:0.0"
    "light:0.005:0.5:2.0"
    "medium:0.01:2.0:10.0" 
    "heavy:0.05:10.0:50.0"
)
```

## 📈 Analysis and Visualization

### Automatic Analysis

The system includes automatic analysis tools:

```bash
# Run analysis after grid search completes
python gridsearch/analyze_grid_search_results.py
```

### Manual Analysis

Key metrics to examine:

1. **Final Performance**: Average reward in final episodes
2. **Sample Efficiency**: Episodes needed to reach target performance
3. **Stability**: Variance in performance across seeds
4. **Convergence**: Training curves and convergence patterns

### Performance Metrics

- **Final Reward**: Ultimate performance achieved
- **Convergence Episode**: When stable performance was reached
- **Training Time**: Computational efficiency
- **Success Rate**: Percentage of successful experiment completions

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch sizes or number of parallel environments
2. **Slow Training**: Check GPU utilization and reduce experiment scope
3. **Failed Experiments**: Check individual log files for error details
4. **Inconsistent Results**: Increase number of seeds for statistical significance

### Performance Optimization

```bash
# For faster experimentation (testing setup)
QUICK_TEST=true              # Use minimal parameter space
NUM_EPOCHS=50               # Reduce training time
SEEDS=(0)                   # Single seed for quick testing

# For production runs
NUM_EPOCHS=500              # Full training
SEEDS=(0 42 123 456 789)    # Multiple seeds for significance
EXPERIMENT_DELAY=0          # No delay between experiments
```

### Resource Management

```bash
# Adjust based on your system
NUM_ENV_WORKERS=8           # Parallel environment workers
NUM_CPUS=12                # Total CPU cores
NUM_GPUS=0.1               # GPU allocation per experiment
EXPERIMENT_DELAY=2         # Seconds between experiments
```

## 📋 Best Practices

### Experimental Design

1. **Start Small**: Begin with a subset of parameters to validate the setup
2. **Multiple Seeds**: Use at least 3 different random seeds
3. **Baseline First**: Always include known good configurations as baselines
4. **Statistical Significance**: Plan for multiple runs per configuration

### Parameter Selection

1. **Learning Rate**: Start with 3e-4, explore 1e-4 to 1e-3
2. **Discount Factor**: Use 0.99 as baseline, test 0.95-0.999 range
3. **Exploration**: Start with 0.05 entropy coefficient
4. **Reward Scale**: Keep total rewards in 1-20 range for stability

### Computational Efficiency

1. **Parallel Experiments**: Run multiple experiments simultaneously if resources allow
2. **Early Stopping**: Implement early stopping for clearly failing experiments
3. **Checkpoint Saving**: Save intermediate checkpoints for long experiments
4. **Resource Monitoring**: Monitor CPU/GPU usage to optimize resource allocation

## 🔧 Customization

### Adding New Parameters

To add new hyperparameters to the grid search:

1. **Modify training-DTDE-gridsearch.py**: Add new command-line arguments
2. **Update grid_search_config.sh**: Add parameter ranges
3. **Modify launch scripts**: Include new parameters in experiment loops

### Custom Analysis

Create domain-specific analysis by modifying `analyze_grid_search_results.py`:

```python
def custom_analysis(df):
    # Add your specific analysis logic
    # Example: correlation between entropy coefficient and final reward
    correlation = df['ent_coef'].corr(df['final_reward'])
    print(f"Entropy-Performance Correlation: {correlation:.3f}")
```

## 📖 Example Usage Scenarios

### Scenario 1: Quick Validation Run

```bash
# Edit grid_search_config.sh
QUICK_TEST=true
LR_VALUES=(0.0003)
SEEDS=(0)
PENALTY_CONFIGS=("medium:0.01:2.0:10.0")
REWARD_CONFIGS=("shaped:2.0:5.0:10.0")

# Run
./launch_grid_search.sh
```

### Scenario 2: Comprehensive Hyperparameter Study

```bash
# Full exploration of key hyperparameters
LR_VALUES=(0.0001 0.0003 0.001)
SEEDS=(0 42 123)
GAMMA_VALUES=(0.95 0.99 0.995)
GAE_LAMBDA_VALUES=(0.9 0.95 0.99)
ENT_COEF_VALUES=(0.01 0.05 0.1)
```

### Scenario 3: Reward Engineering Focus

```bash
# Keep hyperparameters fixed, explore reward configurations
LR_VALUES=(0.0003)
GAMMA_VALUES=(0.99)
# ... fix other hyperparameters

REWARD_CONFIGS=(
    "sparse:0.0:0.0:10.0"
    "cut_focused:5.0:2.0:10.0"
    "salad_focused:2.0:8.0:10.0"
    "delivery_focused:1.0:1.0:20.0"
    "equal:5.0:5.0:5.0"
)
```

## 🎓 Interpreting Results

### What to Look For

1. **Convergence Patterns**: How quickly do different configurations converge?
2. **Final Performance**: Which configurations achieve the highest rewards?
3. **Stability**: Which configurations are most consistent across seeds?
4. **Efficiency**: Which configurations learn fastest (sample efficiency)?

### Red Flags

- **No Convergence**: Very low or flat learning curves
- **High Variance**: Large differences between seeds for same configuration
- **Reward Hacking**: High rewards but poor actual performance
- **Training Instability**: Erratic learning curves or crashes

## 📚 Additional Resources

- **PPO Algorithm**: [Proximal Policy Optimization paper](https://arxiv.org/abs/1707.06347)
- **Reward Shaping**: [Potential-based reward shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
- **Hyperparameter Tuning**: Best practices for RL hyperparameter optimization

---

**Happy Experimenting! 🚀**

For questions or issues, check the individual experiment logs in the `grid_search_logs` directory or consult the summary files generated after each run.