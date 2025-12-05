# Grid Search Configuration File
# This file defines the parameter ranges and configurations for the grid search experiment
# Edit this file to customize your grid search without modifying the main script

# Base Configuration
# ==================
CLUSTER="cuenca"
INPUT_PATH="input_1.0_1.0.txt"
MAP_NR="baseline_division_of_labor_v2"
GAME_VERSION="classic"
NUM_AGENTS=1
NUM_EPOCHS=50
CHECKPOINT_PATHS="none"
REWARDS_ONLY_ON_DELIVERY="false"
AGENT_TO_TRAIN=1

# Learning Rate Values
# ====================
# Explore different learning rates for optimization
LR_VALUES=(0.0001 0.0003 0.001)

# Random Seeds
# ============
# Multiple seeds for statistical significance (recommend at least 3)
SEEDS=(0 42)

# PPO Hyperparameters
# ===================
# Discount factor (gamma) - higher values emphasize long-term rewards
GAMMA_VALUES=(0.9 0.95 0.99)

# GAE Lambda - bias-variance tradeoff in advantage estimation
GAE_LAMBDA_VALUES=(0.9 0.95 0.99)

# Entropy coefficient - controls exploration vs exploitation
ENT_COEF_VALUES=(0.01 0.05 0.1)

# PPO clip parameter - limits policy updates for stability
CLIP_EPS_VALUES=(0.1 0.2 0.3)

# Value function coefficient - relative weight of value loss
VF_COEF_VALUES=(0.25 0.5 1.0)

# Training Batch Hyperparameters
# ===============================
# Training batch size - samples per training iteration
# Note: Must be compatible with num_env_runners=8, num_envs_per_env_runner=1, rollout_fragment_length=500
# This gives us 8×1×500=4000 samples per rollout, so batch sizes should be multiples of 4000 or factors
TRAIN_BATCH_SIZE_VALUES=(4000)

# SGD minibatch size - samples per SGD update
# Should be factors of the training batch sizes above
SGD_MINIBATCH_SIZE_VALUES=(500)

# Number of SGD iterations per training batch
NUM_SGD_ITER_VALUES=(10)

# Neural Network Architecture
# ===========================
# MLP hidden layer configurations
# Format: "name:layer1:layer2:layer3"
MLP_ARCHITECTURES=(
    "small:256:128:64"
    "medium:512:512:256"  
    "large:1024:512:256"
)

# Penalty Configurations
# ======================
# Format: "name:busy_penalty:useless_action_penalty:destructive_action_penalty"
# 
# busy_penalty: Penalty per second spent busy (encourage efficiency)
# useless_action_penalty: Penalty for actions that don't contribute to goals
# destructive_action_penalty: Penalty for actions that undo progress
PENALTY_CONFIGS=(
    "low:0.005:0.5:2.0"      # Lenient penalties - allows more exploration
    "high:0.01:2.0:10.0"     # Strict penalties - enforces efficient behavior
)

# Reward Configurations
# =====================
# Format: "name:cut_reward:salad_reward:deliver_reward"
#
# cut_reward: Reward for cutting ingredients
# salad_reward: Reward for making salads
# deliver_reward: Reward for delivering completed orders
REWARD_CONFIGS=(
    "only_final:0.0:0.0:10.0"      # Only reward final delivery - sparse rewards
    "guided:2.0:5.0:10.0"      # Intermediate rewards to guide learning
)

# Experimental Configurations
# ===========================
# You can define specific combinations to test particular hypotheses

# Exploration vs Exploitation Study
#EXPLORATION_CONFIGS=(
#    "high_exploration:0.99:0.99:0.1:0.3:0.5"    # High entropy, high clip
#    "balanced:0.99:0.95:0.05:0.2:0.5"           # Balanced exploration
#    "low_exploration:0.99:0.9:0.01:0.1:0.5"     # Low entropy, low clip
#)

# Temporal Preference Study  
#TEMPORAL_CONFIGS=(
#    "short_term:0.9:0.9"     # Low gamma and lambda - focus on immediate rewards
#    "medium_term:0.95:0.95"  # Medium temporal horizon
#    "long_term:0.995:0.99"   # High gamma and lambda - focus on long-term rewards
#)

# Advanced Configurations
# =======================
# Enable these for more comprehensive searches (will significantly increase experiment count)

# Network Architecture Variants (optional - requires code modification)
# MLP_ARCHITECTURES=(
#     "small:256,128"
#     "medium:512,256"
#     "large:512,512,256"
#     "deep:256,256,256,128"
# )

# Training Schedule Variants (optional)
# TRAINING_SCHEDULES=(
#     "short:200"
#     "medium:300" 
#     "long:500"
# )

# Batch Size Variants (optional)
# BATCH_SIZES=(
#     "small:2000"
#     "medium:4000"
#     "large:8000"
# )

# Quick Test Configuration
# ========================
# Use this for testing the grid search setup with minimal experiments

QUICK_TEST=false  # Set to true for quick testing

if [ "$QUICK_TEST" = true ]; then
    echo "Running in QUICK TEST mode - limited parameter space"
    LR_VALUES=(0.0003)
    SEEDS=(0)
    GAMMA_VALUES=(0.99)
    GAE_LAMBDA_VALUES=(0.95)
    ENT_COEF_VALUES=(0.05)
    CLIP_EPS_VALUES=(0.2)
    VF_COEF_VALUES=(0.5)
    TRAIN_BATCH_SIZE_VALUES=(4000)  # Compatible with 8×1×500=4000 rollout
    SGD_MINIBATCH_SIZE_VALUES=(500)
    NUM_SGD_ITER_VALUES=(10)
    MLP_ARCHITECTURES=("medium:512:256:128")
    PENALTY_CONFIGS=("medium:0.01:2.0:10.0")
    REWARD_CONFIGS=("shaped:2.0:5.0:10.0")
fi

# Output Configuration
# ====================
# Experiment naming and organization
EXPERIMENT_BASE_NAME="dtde_FIRST_gridsearch"
INCLUDE_TIMESTAMP=true

# Logging and Monitoring
VERBOSE_LOGGING=true

# Resource Management
# ===================
# Delay between experiments to prevent system overload
EXPERIMENT_DELAY=15  # seconds

# Parallel execution
MAX_PARALLEL_EXPERIMENTS=25  # Number of experiments to run in parallel

# Analysis Configuration
# ======================
# Automatic analysis after completion
RUN_ANALYSIS=true
CREATE_PLOTS=true
GENERATE_REPORT=true

# Performance Metrics to Track
# ============================
TRACK_METRICS=(
    "final_reward"
    "convergence_episode" 
    "training_time"
    "sample_efficiency"
    "policy_entropy"
    "value_loss"
)

# Early Stopping Configuration
# ============================
# Stop experiments that are clearly not performing well
ENABLE_EARLY_STOPPING=false
EARLY_STOP_THRESHOLD=0.1  # Minimum reward threshold
EARLY_STOP_EPISODES=1000    # Episodes to evaluate before stopping

echo "Grid search configuration loaded successfully"
echo "Experiment will test ${#LR_VALUES[@]} learning rates, ${#SEEDS[@]} seeds,"
echo "${#GAMMA_VALUES[@]} gamma values, ${#GAE_LAMBDA_VALUES[@]} GAE lambda values,"
echo "${#ENT_COEF_VALUES[@]} entropy coefficients, ${#CLIP_EPS_VALUES[@]} clip values,"
echo "${#VF_COEF_VALUES[@]} value function coefficients,"
echo "${#TRAIN_BATCH_SIZE_VALUES[@]} batch sizes, ${#SGD_MINIBATCH_SIZE_VALUES[@]} minibatch sizes,"
echo "${#NUM_SGD_ITER_VALUES[@]} SGD iterations, ${#MLP_ARCHITECTURES[@]} MLP architectures,"
echo "${#PENALTY_CONFIGS[@]} penalty configurations, and ${#REWARD_CONFIGS[@]} reward configurations"

total_configs=$((${#LR_VALUES[@]} * ${#SEEDS[@]} * ${#GAMMA_VALUES[@]} * ${#GAE_LAMBDA_VALUES[@]} * ${#ENT_COEF_VALUES[@]} * ${#CLIP_EPS_VALUES[@]} * ${#VF_COEF_VALUES[@]} * ${#TRAIN_BATCH_SIZE_VALUES[@]} * ${#SGD_MINIBATCH_SIZE_VALUES[@]} * ${#NUM_SGD_ITER_VALUES[@]} * ${#MLP_ARCHITECTURES[@]} * ${#PENALTY_CONFIGS[@]} * ${#REWARD_CONFIGS[@]}))
echo "Total configurations: $total_configs"