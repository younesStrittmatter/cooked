"""
Ray cluster management utilities.

Author: Samuel Lozano
"""

import ray


class RayManager:
    """Manages Ray cluster initialization and shutdown."""
    
    @staticmethod
    def initialize_ray():
        """Initialize Ray in local mode."""
        try:
            ray.shutdown()
            ray.init(local_mode=True, ignore_reinit_error=True, include_dashboard=False)
            print("Ray initialized successfully")
        except Exception as e:
            print(f"Warning: Ray initialization failed: {e}")
    
    @staticmethod
    def shutdown_ray():
        """Shutdown Ray cluster."""
        try:
            ray.shutdown()
            print("Ray shutdown successfully")
        except Exception as e:
            print(f"Warning: Ray shutdown failed: {e}")