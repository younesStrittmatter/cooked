# Analysis System for Reinforcement Learning Experiments

This directory contains a comprehensive and professional analysis system for reinforcement learning experiments. The system has been refactored to eliminate code duplication, improve maintainability, and provide a consistent interface across different experiment types.

## Features

- **Modular Design**: Common functionality is shared across all analysis scripts
- **Command-line Interface**: Easy parameter configuration via command-line arguments
- **Professional Code Structure**: Clean, documented, and maintainable code
- **Flexible Configuration**: Support for different clusters and experiment types
- **Comprehensive Visualization**: Automated generation of analysis plots

## Quick Start

### Basic Usage

```bash
# Classic experiments
python analysis_classic_new.py v3.1 simple_kitchen_circular

# Competition experiments  
python analysis_competition_new.py v3.1 simple_kitchen_competition

# Pretrained experiments
python analysis_pretrained_new.py v3.1 simple_kitchen_circular
```

### Advanced Usage

```bash
# Specify cluster and smoothing factor
python analysis_classic_new.py v3.1 simple_kitchen_circular --cluster cuenca --smoothing-factor 20

# Use different output format
python analysis_competition_new.py v3.1 simple_kitchen_competition --output-format pdf

# Local analysis
python analysis_pretrained_new.py v3.1 simple_kitchen_circular --cluster local
```

## Command-line Options

All analysis scripts support the following options:

- `intent_version`: Intent version identifier (required)
- `map_name`: Map name identifier (required)
- `--cluster`: Cluster type (`brigit`, `cuenca`, `local`) [default: cuenca]
- `--smoothing-factor`: Smoothing factor for curves [default: 15]
- `--output-format`: Output format for figures (`png`, `pdf`, `svg`) [default: png]

## Architecture

### Core Components

1. **`utils.py`**: Central utility module containing:
   - `AnalysisConfig`: Configuration management
   - `DataProcessor`: Data loading and preprocessing
   - `MetricDefinitions`: Metric definitions and visual properties
   - `PlotGenerator`: Plot generation utilities
   - `main_analysis_pipeline`: Shared analysis pipeline

2. **Analysis Scripts**:
   - `analysis_classic_new.py`: Single-agent classic experiments
   - `analysis_competition_new.py`: Two-agent competitive experiments
   - `analysis_pretrained_new.py`: Pretrained model experiments

### Directory Structure

```
spoiled_broth/analysis/
├── __init__.py          # Module initialization
├── utils.py             # Core utilities and shared functions
└── README.md            # This documentation

Root directory:
├── analysis_classic_new.py      # Classic experiment analysis
├── analysis_competition_new.py  # Competition experiment analysis
└── analysis_pretrained_new.py   # Pretrained experiment analysis
```

## Key Improvements

### 1. Code Reduction and Deduplication
- **Before**: ~3000 lines across 3 files with significant duplication
- **After**: ~1000 lines with shared utilities, eliminating redundancy

### 2. Professional Structure
- Clean separation of concerns
- Comprehensive documentation
- Type hints for better code clarity
- Error handling and validation

### 3. Enhanced Flexibility
- Command-line parameter configuration
- Configurable cluster paths
- Modular metric definitions
- Extensible plotting system

### 4. Improved Maintainability
- Single source of truth for common functionality
- Consistent naming conventions
- Modular design for easy extension
- Clear separation between experiment types

## Output

The analysis generates:

1. **CSV File**: `training_results.csv` with consolidated data
2. **Figure Directories**:
   - `figures/`: Standard plots
   - `figures/smoothed_N/`: Smoothed plots (where N is the smoothing factor)

### Generated Plots

- Score/reward vs epoch plots
- Individual agent performance metrics
- Attitude-specific analysis
- Smoothed trend visualizations
- Comparative analysis between game types

## Migration Guide

### From Old to New Scripts

**Old Usage:**
```bash
# Required manual editing of script variables
python analysis_classic.py  # Fixed parameters in code
```

**New Usage:**
```bash
# Flexible command-line interface
python analysis_classic_new.py v3.1 simple_kitchen_circular --cluster cuenca
```

### Key Changes

1. **Parameters**: Now passed via command line instead of hardcoded
2. **Imports**: Use shared utilities from `spoiled_broth.analysis.utils`
3. **Structure**: Experiment-specific logic separated from common functionality
4. **Paths**: Automatic path configuration based on cluster selection

## Extension Guide

### Adding New Experiment Types

1. Create a new analysis script following the pattern of existing scripts
2. Add experiment-specific metrics to `MetricDefinitions`
3. Implement custom plotting logic using the shared `PlotGenerator`
4. Use the common `main_analysis_pipeline` for data loading

### Adding New Metrics

1. Add metric definitions to `MetricDefinitions.get_X_metrics()`
2. Add labels to `MetricDefinitions.get_metric_labels()`
3. Add colors to `MetricDefinitions.get_metric_colors()`
4. Use existing plotting functions or extend `PlotGenerator`

### Customizing Plots

The `PlotGenerator` class provides flexible plotting utilities that can be extended or customized for specific needs. Override methods or create new plotting functions as needed.

## Error Handling

The system includes comprehensive error handling:

- Invalid cluster selection
- Missing data files
- Parsing errors
- Directory creation issues
- File I/O problems

All errors provide clear messages for debugging and resolution.

## Performance

The refactored system is optimized for:

- **Memory efficiency**: Streaming data processing
- **Speed**: Vectorized operations with pandas/numpy
- **Scalability**: Modular design supports large datasets

## Contributing

When modifying the analysis system:

1. Follow the established code structure
2. Add appropriate documentation
3. Include type hints
4. Test with all experiment types
5. Update this README if adding new features

## Dependencies

- pandas: Data manipulation and analysis
- matplotlib: Plotting and visualization
- numpy: Numerical computations
- argparse: Command-line argument parsing
- typing: Type hints support