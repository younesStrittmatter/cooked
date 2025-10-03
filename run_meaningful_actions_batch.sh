#!/bin/bash

# Script to run meaningful_actions.py over all trainings and simulations
# for checkpoint = final and a defined map
#
# Usage: nohup ./run_meaningful_actions_batch.sh <map_name> [cluster] > log_file 2>&1 &
#   map_name: The map to analyze (e.g., "baseline_division_of_labor")
#   cluster: Optional cluster name (brigit, cuenca, local). Default: cuenca
#
# Example: nohup ./run_meaningful_actions_batch.sh baseline_division_of_labor cuenca > log_run_meaningful_actions.out 2>&1 &

# Check if map argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <map_name> [cluster]"
    echo "  map_name: The map to analyze (e.g., 'baseline_division_of_labor')"
    echo "  cluster: Optional cluster name (brigit, cuenca, local). Default: cuenca"
    exit 1
fi

MAP_NAME="$1"
CLUSTER="${2:-cuenca}"  # Default to cuenca if not specified
CHECKPOINT="final"

# Configuration for different clusters
case "$CLUSTER" in
    "brigit")
        LOCAL="/mnt/lustre/home/samuloza"
        ;;
    "cuenca")
        LOCAL=""
        ;;
    "local")
        LOCAL="D:/OneDrive - Universidad Complutense de Madrid (UCM)/Doctorado"
        ;;
    *)
        echo "Error: Invalid cluster specified. Choose from 'brigit', 'cuenca', or 'local'."
        exit 1
        ;;
esac

# Fixed configuration
GAME_VERSION="classic"
INTENT_VERSION="v3.1"

# Base path for data
BASE_DATA_PATH="${LOCAL}/data/samuel_lozano/cooked/${GAME_VERSION}/${INTENT_VERSION}/map_${MAP_NAME}"

# Script directory (where meaningful_actions.py is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEANINGFUL_ACTIONS_SCRIPT="${SCRIPT_DIR}/spoiled_broth/simulations/meaningful_actions.py"

# Check if meaningful_actions.py exists
if [ ! -f "$MEANINGFUL_ACTIONS_SCRIPT" ]; then
    echo "Error: meaningful_actions.py not found at $MEANINGFUL_ACTIONS_SCRIPT"
    exit 1
fi

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to process a single simulation
process_simulation() {
    local simulation_dir="$1"
    local training_id="$2"
    local cooperation_type="$3"
    local simulation_id="$4"
    
    local actions_csv="${simulation_dir}/actions.csv"
    local simulation_csv="${simulation_dir}/simulation.csv"
    
    # Check if required files exist
    if [ ! -f "$actions_csv" ]; then
        log "WARNING: actions.csv not found in $simulation_dir"
        return 1
    fi
    
    if [ ! -f "$simulation_csv" ]; then
        log "WARNING: simulation.csv not found in $simulation_dir"
        return 1
    fi
    
    log "Processing simulation: $training_id/$cooperation_type/simulations_${CHECKPOINT}/$simulation_id"
    
    # Run meaningful_actions.py
    python3 "$MEANINGFUL_ACTIONS_SCRIPT" \
        --actions_csv "$actions_csv" \
        --simulation_csv "$simulation_csv" \
        --map "$MAP_NAME" \
        --output_dir "$simulation_dir"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log "SUCCESS: Completed simulation $simulation_id"
    else
        log "ERROR: Failed to process simulation $simulation_id (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Function to process all simulations in a training directory
process_training() {
    local training_dir="$1"
    local training_id="$2"
    local cooperation_type="$3"
    
    local simulations_dir="${training_dir}/simulations_${CHECKPOINT}"
    
    if [ ! -d "$simulations_dir" ]; then
        log "WARNING: simulations_${CHECKPOINT} directory not found in $training_dir"
        return 1
    fi
    
    log "Processing training: $training_id ($cooperation_type)"
    
    local simulation_count=0
    local success_count=0
    local error_count=0
    
    # Process each simulation directory
    for simulation_path in "$simulations_dir"/simulation_*; do
        if [ -d "$simulation_path" ]; then
            local simulation_id=$(basename "$simulation_path")
            simulation_count=$((simulation_count + 1))
            
            if process_simulation "$simulation_path" "$training_id" "$cooperation_type" "$simulation_id"; then
                success_count=$((success_count + 1))
            else
                error_count=$((error_count + 1))
            fi
        fi
    done
    
    log "Training $training_id summary: $simulation_count simulations, $success_count successful, $error_count errors"
    return 0
}

# Main execution
main() {
    log "Starting meaningful actions batch analysis"
    log "Map: $MAP_NAME"
    log "Cluster: $CLUSTER"
    log "Checkpoint: $CHECKPOINT"
    log "Base data path: $BASE_DATA_PATH"
    
    # Check if base data path exists
    if [ ! -d "$BASE_DATA_PATH" ]; then
        log "ERROR: Base data path does not exist: $BASE_DATA_PATH"
        exit 1
    fi
    
    local total_trainings=0
    local successful_trainings=0
    
    # Process cooperative trainings
    local cooperative_dir="${BASE_DATA_PATH}/cooperative"
    if [ -d "$cooperative_dir" ]; then
        log "Processing cooperative trainings..."
        for training_path in "$cooperative_dir"/Training_*; do
            if [ -d "$training_path" ]; then
                local training_id=$(basename "$training_path" | sed 's/Training_//')
                total_trainings=$((total_trainings + 1))
                
                if process_training "$training_path" "$training_id" "cooperative"; then
                    successful_trainings=$((successful_trainings + 1))
                fi
            fi
        done
    else
        log "WARNING: Cooperative directory not found: $cooperative_dir"
    fi
    
    # Process competitive trainings
    local competitive_dir="${BASE_DATA_PATH}/competitive"
    if [ -d "$competitive_dir" ]; then
        log "Processing competitive trainings..."
        for training_path in "$competitive_dir"/Training_*; do
            if [ -d "$training_path" ]; then
                local training_id=$(basename "$training_path" | sed 's/Training_//')
                total_trainings=$((total_trainings + 1))
                
                if process_training "$training_path" "$training_id" "competitive"; then
                    successful_trainings=$((successful_trainings + 1))
                fi
            fi
        done
    else
        log "WARNING: Competitive directory not found: $competitive_dir"
    fi
    
    # Final summary
    log "=== BATCH ANALYSIS COMPLETE ==="
    log "Total trainings processed: $total_trainings"
    log "Successful trainings: $successful_trainings"
    log "Failed trainings: $((total_trainings - successful_trainings))"
    
    if [ $total_trainings -eq 0 ]; then
        log "WARNING: No training directories found!"
        exit 1
    fi
}

# Run main function
main "$@"