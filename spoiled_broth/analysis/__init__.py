"""
Analysis module for reinforcement learning experiments.

This module provides utilities for analyzing training results from various
types of RL experiments including classic, competition, and pretrained models.
"""

from .utils import (
    AnalysisConfig,
    DataProcessor,
    MetricDefinitions,
    PlotGenerator,
    setup_argument_parser,
    main_analysis_pipeline
)

__all__ = [
    'AnalysisConfig',
    'DataProcessor', 
    'MetricDefinitions',
    'PlotGenerator',
    'setup_argument_parser',
    'main_analysis_pipeline'
]