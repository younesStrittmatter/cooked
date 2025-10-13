#!/bin/bash

# Script to run meaningful actions testing on all simulation directories
# Usage: nohup ./run_meaningful_actions_testing.sh > testing_meaningful_actions_batch.log 2>&1 &

set -e  # Exit on any error

# Configuration
MAP_NR=baseline_division_of_labor
BASE_DIR="/data/samuel_lozano/cooked/classic/v3.1/map_${MAP_NR}/simulations"
SCRIPT_DIR="/home/samuel_lozano/cooked"
PYTHON_SCRIPT="$SCRIPT_DIR/testing_meaningful_actions.py"
LOG_DIR="$SCRIPT_DIR/logs_testing_meaningful_actions"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "MEANINGFUL ACTIONS TESTING - BATCH RUN"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo "Python script: $PYTHON_SCRIPT"
echo "Log directory: $LOG_DIR"
echo "Started at: $(date)"
echo

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Find all simulation directories that contain actions.csv
echo "Finding simulation directories..."
SIMULATION_DIRS=$(find "$BASE_DIR" -name "actions.csv" -type f | sed 's|/actions.csv$||' | sort)

if [ -z "$SIMULATION_DIRS" ]; then
    echo "ERROR: No simulation directories with actions.csv found in $BASE_DIR"
    exit 1
fi

# Count total directories
TOTAL_DIRS=$(echo "$SIMULATION_DIRS" | wc -l)
echo "Found $TOTAL_DIRS simulation directories to process"
echo

# Initialize counters
CURRENT=0
SUCCESS=0
FAILED=0

# Process each directory
while IFS= read -r sim_dir; do
    CURRENT=$((CURRENT + 1))
    
    echo "[$CURRENT/$TOTAL_DIRS] Processing: $sim_dir"
    
    # Extract training_id and simulation_id from path
    # Path format: .../Training_YYYY-MM-DD_HH-MM-SS/checkpoint_N/simulation_YYYY_MM_DD-HH_MM_SS
    
    # Extract training timestamp (Training_YYYY-MM-DD_HH-MM-SS)
    TRAINING_ID=$(echo "$sim_dir" | grep -o 'Training_[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}' | sed 's/Training_//')
    
    # Extract checkpoint identifier (checkpoint_N or checkpoint_final)
    CHECKPOINT_ID=$(echo "$sim_dir" | grep -o 'checkpoint_[0-9a-z_]\+' | sed 's/checkpoint_//')
    
    # Extract simulation timestamp (simulation_YYYY_MM_DD-HH_MM_SS)
    SIMULATION_ID=$(echo "$sim_dir" | grep -o 'simulation_[0-9]\{4\}_[0-9]\{2\}_[0-9]\{2\}-[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}' | sed 's/simulation_//')
    
    if [ -z "$TRAINING_ID" ] || [ -z "$SIMULATION_ID" ]; then
        echo "  ERROR: Could not extract training_id or simulation_id from path"
        echo "  Path: $sim_dir"
        echo "  Training ID: $TRAINING_ID"
        echo "  Checkpoint ID: $CHECKPOINT_ID"
        echo "  Simulation ID: $SIMULATION_ID"
        FAILED=$((FAILED + 1))
        echo
        continue
    fi
    
    # Create log filename
    LOG_FILE="$LOG_DIR/log_${MAP_NR}_${TRAINING_ID}_${CHECKPOINT_ID}_${SIMULATION_ID}.out"
    
    echo "  Training ID: $TRAINING_ID"
    echo "  Checkpoint ID: $CHECKPOINT_ID"
    echo "  Simulation ID: $SIMULATION_ID"
    echo "  Log file: $LOG_FILE"
    
    # Check if actions.csv and simulation.csv exist
    if [ ! -f "$sim_dir/actions.csv" ]; then
        echo "  ERROR: actions.csv not found in $sim_dir"
        FAILED=$((FAILED + 1))
        echo
        continue
    fi
    
    if [ ! -f "$sim_dir/simulation.csv" ]; then
        echo "  ERROR: simulation.csv not found in $sim_dir"
        FAILED=$((FAILED + 1))
        echo
        continue
    fi
    
    # Run the testing script
    echo "  Running analysis..."
    START_TIME=$(date +%s)
    
    if python3 "$PYTHON_SCRIPT" --path "$sim_dir" > "$LOG_FILE" 2>&1; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "  ✅ SUCCESS (${DURATION}s)"
        SUCCESS=$((SUCCESS + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "  ❌ FAILED (${DURATION}s) - check log: $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
    
    echo
done <<< "$SIMULATION_DIRS"

# Final summary
echo "=========================================="
echo "BATCH RUN COMPLETED"
echo "=========================================="
echo "Total directories processed: $TOTAL_DIRS"
echo "Successful analyses: $SUCCESS"
echo "Failed analyses: $FAILED"
echo "Success rate: $(( SUCCESS * 100 / TOTAL_DIRS ))%"
echo "Completed at: $(date)"
echo
echo "Log files saved in: $LOG_DIR"
echo "Log filename pattern: testing_meaningful_actions_{training_id}_{simulation_id}.out"

if [ $FAILED -gt 0 ]; then
    echo
    echo "⚠️  Some analyses failed. Check the log files for details."
    exit 1
else
    echo
    echo "🎉 All analyses completed successfully!"
fi