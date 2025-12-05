#!/bin/bash

# Advanced Grid Search Script for DTDE Training
# This script systematically explores different hyperparameters and reward configurations
# Uses external configuration file for easy customization

# Forzar el uso de punto decimal en printf
LC_NUMERIC=en_US.UTF-8

# Source configuration file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/grid_search_config.sh"

if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "Configuration file not found: $CONFIG_FILE"
    echo "Creating default configuration..."
    
    # Fallback to basic configuration
    CLUSTER="cuenca"
    INPUT_PATH="input_1.0_1.0.txt"
    MAP_NR="baseline_division_of_labor_v2"
    GAME_VERSION="classic"
    NUM_AGENTS=1
    NUM_EPOCHS=300
    CHECKPOINT_PATHS="none"
    REWARDS_ONLY_ON_DELIVERY="false"
    AGENT_TO_TRAIN=1
    
    LR_VALUES=(0.0003)
    SEEDS=(0)
    GAMMA_VALUES=(0.99)
    GAE_LAMBDA_VALUES=(0.95)
    ENT_COEF_VALUES=(0.05)
    CLIP_EPS_VALUES=(0.2)
    VF_COEF_VALUES=(0.5)
    PENALTY_CONFIGS=("medium:0.01:2.0:10.0")
    REWARD_CONFIGS=("shaped:2.0:5.0:10.0")
    
    EXPERIMENT_DELAY=30
    MAX_PARALLEL_EXPERIMENTS=4
fi

echo "Starting DTDE Grid Search Experiment"
echo "====================================="

# Create experiment directory with timestamp
if [ "$INCLUDE_TIMESTAMP" = true ]; then
    EXPERIMENT_NAME="${EXPERIMENT_BASE_NAME:-gridsearch}_$(date +%Y-%m-%d_%H-%M-%S)"
else
    EXPERIMENT_NAME="${EXPERIMENT_BASE_NAME:-gridsearch}"
fi

mkdir -p "/data/samuel_lozano/cooked/gridsearch/$EXPERIMENT_NAME"
LOG_DIR="/data/samuel_lozano/cooked/gridsearch/$EXPERIMENT_NAME"

# Create summary file
SUMMARY_FILE="$LOG_DIR/grid_search_summary.txt"
echo "Grid Search Summary - Started: $(date)" > $SUMMARY_FILE
echo "=====================================" >> $SUMMARY_FILE
echo "Configuration loaded from: $CONFIG_FILE" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Log configuration details
{
    echo "Base Configuration:"
    echo "  Cluster: $CLUSTER"
    echo "  Map: $MAP_NR"
    echo "  Game Version: $GAME_VERSION"
    echo "  Agents: $NUM_AGENTS"
    echo "  Epochs: $NUM_EPOCHS"
    echo ""
    echo "Parameter Ranges:"
    echo "  Learning Rates: ${LR_VALUES[@]}"
    echo "  Seeds: ${SEEDS[@]}"
    echo "  Gamma Values: ${GAMMA_VALUES[@]}"
    echo "  GAE Lambda Values: ${GAE_LAMBDA_VALUES[@]}"
    echo "  Entropy Coefficients: ${ENT_COEF_VALUES[@]}"
    echo "  Clip Values: ${CLIP_EPS_VALUES[@]}"
    echo "  Value Function Coefficients: ${VF_COEF_VALUES[@]}"
    echo ""
    echo "Penalty Configurations:"
    for config in "${PENALTY_CONFIGS[@]}"; do
        echo "  $config"
    done
    echo ""
    echo "Reward Configurations:"
    for config in "${REWARD_CONFIGS[@]}"; do
        echo "  $config"
    done
    echo ""
} >> $SUMMARY_FILE

# Function to log experiment details
log_experiment() {
    local config_name="$1"
    local lr="$2"
    local seed="$3"
    local gamma="$4"
    local gae_lambda="$5"
    local ent_coef="$6"
    local clip_eps="$7"
    local vf_coef="$8"
    local penalty_config="$9"
    local reward_config="${10}"
    
    {
        echo "Configuration: $config_name"
        echo "  LR: $lr, SEED: $seed"
        echo "  Hyperparams: γ=$gamma, λ=$gae_lambda, ent=$ent_coef, clip=$clip_eps, vf=$vf_coef"
        echo "  Penalties: $penalty_config"
        echo "  Rewards: $reward_config"
        echo "  Started: $(date)"
        echo ""
    } >> $SUMMARY_FILE
}

# Array to store process IDs
PIDS=()
RUNNING_COUNT=0

# Function to wait for a job slot to become available
wait_for_slot() {
    while [ "$RUNNING_COUNT" -ge "${MAX_PARALLEL_EXPERIMENTS:-4}" ]; do
        echo "Max parallel limit (${MAX_PARALLEL_EXPERIMENTS:-4}) reached. Waiting for a job to finish..."
        # Wait for any background job to finish
        wait -n
        # Update the running count by checking the job table for running jobs
        RUNNING_COUNT=$(jobs -p | wc -l)
        echo "A job completed. $RUNNING_COUNT jobs still running. Resuming launch..."
    done
}

# Function to check if experiment should be stopped early
check_early_stopping() {
    local log_file="$1"
    local episodes_check="$2"
    
    if [ "$ENABLE_EARLY_STOPPING" = true ] && [ -f "$log_file" ]; then
        # Simple check for consistently low rewards (you may need to adjust based on your log format)
        local low_reward_count=$(grep -o "reward.*${EARLY_STOP_THRESHOLD}" "$log_file" | wc -l)
        if [ "$low_reward_count" -gt "$episodes_check" ]; then
            return 0  # Early stop
        fi
    fi
    return 1  # Continue
}

# Function to run a single experiment
run_experiment() {
    local config_name="$1"
    local lr="$2"
    local seed="$3"
    local gamma="$4"
    local gae_lambda="$5"
    local ent_coef="$6"
    local clip_eps="$7"
    local vf_coef="$8"
    local busy_penalty="$9"
    local useless_penalty="${10}"
    local destructive_penalty="${11}"
    local cut_reward="${12}"
    local salad_reward="${13}"
    local deliver_reward="${14}"
    local train_batch_size="${15}"
    local sgd_minibatch_size="${16}"
    local num_sgd_iter="${17}"
    local mlp_hidden1="${18}"
    local mlp_hidden2="${19}"
    local mlp_hidden3="${20}"
    local penalty_name="${21}"
    local reward_name="${22}"
    local arch_name="${23}"
    
    # Create unique experiment identifier
    local exp_id="${config_name}_lr${lr}_seed${seed}_batch${train_batch_size}_${arch_name}_${penalty_name}_${reward_name}"
    local output_file="$LOG_DIR/output_${exp_id}.txt"
    local progress_file="$LOG_DIR/progress_${exp_id}.txt"
    
    echo "Starting experiment: $exp_id"
    
    if [ "$VERBOSE_LOGGING" = true ]; then
        echo "Command: python ./training-DTDE-gridsearch.py $CLUSTER $INPUT_PATH $MAP_NR $lr $GAME_VERSION $NUM_AGENTS $NUM_EPOCHS $seed $CHECKPOINT_PATHS $REWARDS_ONLY_ON_DELIVERY $AGENT_TO_TRAIN $gamma $gae_lambda $ent_coef $clip_eps $vf_coef $busy_penalty $useless_penalty $destructive_penalty $cut_reward $salad_reward $deliver_reward $train_batch_size $sgd_minibatch_size $num_sgd_iter $mlp_hidden1 $mlp_hidden2 $mlp_hidden3 $exp_id"
    fi
    
    # Log experiment details
    log_experiment "$config_name" "$lr" "$seed" "$gamma" "$gae_lambda" "$ent_coef" "$clip_eps" "$vf_coef" "${penalty_name}:${busy_penalty}:${useless_penalty}:${destructive_penalty}" "${reward_name}:${cut_reward}:${salad_reward}:${deliver_reward}"
    
    # Create progress tracking
    echo "Experiment: $exp_id" > "$progress_file"
    echo "Started: $(date)" >> "$progress_file"
    echo "Status: Running" >> "$progress_file"
    
    # Run the training in background
    python /home/samuel_lozano/cooked/training-DTDE-gridsearch.py \
        $CLUSTER $INPUT_PATH $MAP_NR $lr $GAME_VERSION $NUM_AGENTS $NUM_EPOCHS $seed \
        $CHECKPOINT_PATHS $REWARDS_ONLY_ON_DELIVERY $AGENT_TO_TRAIN \
        $gamma $gae_lambda $ent_coef $clip_eps $vf_coef \
        $busy_penalty $useless_penalty $destructive_penalty \
        $cut_reward $salad_reward $deliver_reward \
        $train_batch_size $sgd_minibatch_size $num_sgd_iter \
        $mlp_hidden1 $mlp_hidden2 $mlp_hidden3 \
        $exp_id > "$output_file" 2>&1 &
    
    local pid=$!
    echo "PID: $pid" >> "$progress_file"
    echo "Started in background: $(date)" >> "$progress_file"
    
    # Job started in background, no need to return PID
    return 0
}

# Calculate total number of experiments
total_experiments=0
for lr in "${LR_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for gamma in "${GAMMA_VALUES[@]}"; do
            for gae_lambda in "${GAE_LAMBDA_VALUES[@]}"; do
                for ent_coef in "${ENT_COEF_VALUES[@]}"; do
                    for clip_eps in "${CLIP_EPS_VALUES[@]}"; do
                        for vf_coef in "${VF_COEF_VALUES[@]}"; do
                            for train_batch_size in "${TRAIN_BATCH_SIZE_VALUES[@]}"; do
                                for sgd_minibatch_size in "${SGD_MINIBATCH_SIZE_VALUES[@]}"; do
                                    for num_sgd_iter in "${NUM_SGD_ITER_VALUES[@]}"; do
                                        for mlp_arch in "${MLP_ARCHITECTURES[@]}"; do
                                            for penalty_config in "${PENALTY_CONFIGS[@]}"; do
                                                for reward_config in "${REWARD_CONFIGS[@]}"; do
                                                    ((total_experiments++))
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Total experiments to run: $total_experiments"
echo "Results will be saved in: $LOG_DIR"
echo ""

# Add total count to summary
echo "Total experiments planned: $total_experiments" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Main parallel grid search loop
experiment_counter=0
successful_experiments=0
failed_experiments=0
early_stopped_experiments=0

start_time=$(date +%s)

echo "Running experiments with max ${MAX_PARALLEL_EXPERIMENTS:-4} parallel jobs"
echo "Delay between starts: ${EXPERIMENT_DELAY:-30} seconds"
echo ""

for lr in "${LR_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for gamma in "${GAMMA_VALUES[@]}"; do
            for gae_lambda in "${GAE_LAMBDA_VALUES[@]}"; do
                for ent_coef in "${ENT_COEF_VALUES[@]}"; do
                    for clip_eps in "${CLIP_EPS_VALUES[@]}"; do
                        for vf_coef in "${VF_COEF_VALUES[@]}"; do
                            for train_batch_size in "${TRAIN_BATCH_SIZE_VALUES[@]}"; do
                                for sgd_minibatch_size in "${SGD_MINIBATCH_SIZE_VALUES[@]}"; do
                                    for num_sgd_iter in "${NUM_SGD_ITER_VALUES[@]}"; do
                                        for mlp_arch in "${MLP_ARCHITECTURES[@]}"; do
                                            for penalty_config in "${PENALTY_CONFIGS[@]}"; do
                                                for reward_config in "${REWARD_CONFIGS[@]}"; do
                                                    ((experiment_counter++))
                                                    
                                                    # Parse penalty configuration
                                                    IFS=':' read -r penalty_name busy_penalty useless_penalty destructive_penalty <<< "$penalty_config"
                                                    
                                                    # Parse reward configuration
                                                    IFS=':' read -r reward_name cut_reward salad_reward deliver_reward <<< "$reward_config"
                                                    
                                                    # Parse MLP architecture
                                                    IFS=':' read -r arch_name mlp_hidden1 mlp_hidden2 mlp_hidden3 <<< "$mlp_arch"
                                                    
                                                    # Create configuration name
                                                    config_name="exp${experiment_counter}"
                                                    
                                                    # Wait for available slot
                                                    wait_for_slot
                                                    
                                                    # Progress reporting
                                                    current_time=$(date +%s)
                                                    elapsed_time=$((current_time - start_time))
                                                    completed_jobs=$((successful_experiments + failed_experiments + early_stopped_experiments))
                                                    
                                                    if [ $completed_jobs -gt 0 ]; then
                                                        avg_time_per_exp=$((elapsed_time / completed_jobs))
                                                        remaining_experiments=$((total_experiments - completed_jobs))
                                                        estimated_remaining_time=$((avg_time_per_exp * remaining_experiments))
                                                        estimated_completion=$(date -d "@$((current_time + estimated_remaining_time))" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "Unknown")
                                                        
                                                        echo "Progress: Started $experiment_counter/$total_experiments | Completed: $completed_jobs ($(( completed_jobs * 100 / total_experiments ))%)"
                                                        echo "Running: $RUNNING_COUNT/${MAX_PARALLEL_EXPERIMENTS:-4} | Elapsed: $(( elapsed_time / 60 ))m | ETA: $estimated_completion"
                                                    else
                                                        echo "Progress: Started $experiment_counter/$total_experiments | Running: $RUNNING_COUNT/${MAX_PARALLEL_EXPERIMENTS:-4}"
                                                    fi
                                                    
                                                    echo "Starting: $config_name"
                                                    echo "  Training params: batch=$train_batch_size, minibatch=$sgd_minibatch_size, sgd_iter=$num_sgd_iter"
                                                    echo "  Architecture: $arch_name ($mlp_hidden1-$mlp_hidden2-$mlp_hidden3)"
                                                    
                                                    # Start the experiment
                                                    run_experiment "$config_name" "$lr" "$seed" "$gamma" "$gae_lambda" "$ent_coef" "$clip_eps" "$vf_coef" "$busy_penalty" "$useless_penalty" "$destructive_penalty" "$cut_reward" "$salad_reward" "$deliver_reward" "$train_batch_size" "$sgd_minibatch_size" "$num_sgd_iter" "$mlp_hidden1" "$mlp_hidden2" "$mlp_hidden3" "$penalty_name" "$reward_name" "$arch_name"
                                                    
                                                    # Update the running count
                                                    RUNNING_COUNT=$(jobs -p | wc -l)
                                                    
                                                    echo "  → Started in background (Running: $RUNNING_COUNT/${MAX_PARALLEL_EXPERIMENTS:-4})"
                                                    echo ""
                                                    
                                                    # Add delay between experiment starts
                                                    if [ "$EXPERIMENT_DELAY" -gt 0 ] && [ $experiment_counter -lt $total_experiments ]; then
                                                        echo "Waiting ${EXPERIMENT_DELAY}s before next experiment..."
                                                        sleep "$EXPERIMENT_DELAY"
                                                    fi
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# Wait for all remaining jobs to complete
echo "All experiments started. Waiting for remaining jobs to complete..."
echo "Waiting for all background jobs to finish..."

# Wait for ALL remaining background processes to finish
wait

echo "All experiments completed!"

# Final summary
end_time=$(date +%s)
total_time=$((end_time - start_time))

{
    echo ""
    echo "Grid Search Completed: $(date)"
    echo "Total runtime: $(( total_time / 60 )) minutes $(( total_time % 60 )) seconds"
    echo "Total experiments: $total_experiments"
    echo "Successful: $successful_experiments"
    echo "Failed: $failed_experiments"
    echo "Early stopped: $early_stopped_experiments"
    echo "Success rate: $(( successful_experiments * 100 / total_experiments ))%"
} >> $SUMMARY_FILE

echo ""
echo "====================================="
echo "Grid Search Completed!"
echo "Total experiments: $total_experiments"
echo "Successful: $successful_experiments"
echo "Failed: $failed_experiments"
echo "Early stopped: $early_stopped_experiments"
echo "Total runtime: $(( total_time / 60 )) minutes"
echo "Results saved in: $LOG_DIR"
echo "Summary file: $SUMMARY_FILE"
echo "====================================="

# Copy and run analysis if requested
ANALYSIS_SCRIPT_SOURCE="$SCRIPT_DIR/analyze_grid_search_results.py"
ANALYSIS_SCRIPT="$LOG_DIR/analyze_grid_search_results.py"

# Always copy the analysis script to the results directory
if [ -f "$ANALYSIS_SCRIPT_SOURCE" ]; then
    cp "$ANALYSIS_SCRIPT_SOURCE" "$ANALYSIS_SCRIPT"
    chmod +x "$ANALYSIS_SCRIPT"
    echo "Analysis script copied to: $ANALYSIS_SCRIPT"
else
    echo "Warning: Analysis script not found at $ANALYSIS_SCRIPT_SOURCE"
fi

# Run analysis if requested
if [ "$RUN_ANALYSIS" = true ]; then
    echo ""
    echo "Running comprehensive analysis..."
    
    if [ -f "$ANALYSIS_SCRIPT" ]; then
        echo "Using advanced analysis script..."
        python "$ANALYSIS_SCRIPT" "$LOG_DIR" --verbose
        
        if [ $? -eq 0 ]; then
            echo "✓ Analysis completed successfully!"
            echo "Check the following files in $LOG_DIR:"
            echo "  - analysis_report.txt (comprehensive text report)"
            echo "  - grid_search_results.csv (detailed results data)"
            echo "  - *.png (visualization plots)"
        else
            echo "✗ Analysis failed, falling back to basic analysis..."
            RUN_BASIC_ANALYSIS=true
        fi
    else
        echo "Advanced analysis script not available, using basic analysis..."
        RUN_BASIC_ANALYSIS=true
    fi
    
    # Fallback basic analysis
    if [ "$RUN_BASIC_ANALYSIS" = true ]; then
        echo "Creating basic analysis report..."
        
        ANALYSIS_REPORT="$LOG_DIR/basic_analysis_summary.txt"
        echo "Basic Grid Search Analysis Report" > "$ANALYSIS_REPORT"
        echo "Generated: $(date)" >> "$ANALYSIS_REPORT"
        echo "====================================" >> "$ANALYSIS_REPORT"
        echo "" >> "$ANALYSIS_REPORT"
        
        # Count successful experiments by configuration type
        echo "Experiment Results by Configuration:" >> "$ANALYSIS_REPORT"
        for config in "${PENALTY_CONFIGS[@]}"; do
            penalty_name=$(echo "$config" | cut -d':' -f1)
            success_count=$(find "$LOG_DIR" -name "*${penalty_name}*" -type f -name "progress_*" -exec grep -l "Completed Successfully" {} \; 2>/dev/null | wc -l)
            total_count=$(find "$LOG_DIR" -name "*${penalty_name}*" -type f -name "progress_*" 2>/dev/null | wc -l)
            echo "  Penalty $penalty_name: $success_count/$total_count successful" >> "$ANALYSIS_REPORT"
        done
        
        echo "" >> "$ANALYSIS_REPORT"
        for config in "${REWARD_CONFIGS[@]}"; do
            reward_name=$(echo "$config" | cut -d':' -f1)
            success_count=$(find "$LOG_DIR" -name "*${reward_name}*" -type f -name "progress_*" -exec grep -l "Completed Successfully" {} \; 2>/dev/null | wc -l)
            total_count=$(find "$LOG_DIR" -name "*${reward_name}*" -type f -name "progress_*" 2>/dev/null | wc -l)
            echo "  Reward $reward_name: $success_count/$total_count successful" >> "$ANALYSIS_REPORT"
        done
        
        echo "Basic analysis saved to: $ANALYSIS_REPORT"
    fi
else
    echo "Analysis script available at: $ANALYSIS_SCRIPT"
    echo "Run analysis manually with: python $ANALYSIS_SCRIPT $LOG_DIR"
fi

echo ""
echo "Grid search complete. Check the results directory for detailed logs and analysis."