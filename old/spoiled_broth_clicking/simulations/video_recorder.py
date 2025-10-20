"""
Video recording utilities for simulation runs.

Author: Samuel Lozano
"""

import os
from pathlib import Path
from typing import List, Any

import numpy as np
import cv2


class VideoRecorder:
    """Handles video recording with HUD overlay."""
    
    def __init__(self, output_path: Path, fps: int = 24):
        self.output_path = str(output_path)
        self.fps = fps
        self.writer = None
        self.size = None
        self.hud_height = None  # Will be calculated based on first frame
        
        # Create directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
    
    def start(self, frame: np.ndarray):
        """Initialize video writer with first frame."""
        src = self._prepare_frame(frame)
        h, w = src.shape[:2]
        
        # Calculate HUD height (3 lines of text + padding)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        line_height = cv2.getTextSize('Tg', font, scale, thickness)[0][1] + 6
        self.hud_height = max(48, 12 + 3 * line_height)  # 3 lines: Coop, ai_rl_1, ai_rl_2
        
        # Video dimensions include game area + HUD area
        video_width = w
        video_height = h + self.hud_height
        
        # Try different codecs
        fourcc_candidates = ['mp4v', 'avc1', 'H264', 'XVID', 'MJPG']
        
        for code in fourcc_candidates:
            fourcc = cv2.VideoWriter_fourcc(*code)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (video_width, video_height))
            
            try:
                if self.writer.isOpened():
                    self.size = (video_width, video_height)
                    break
            except Exception:
                pass
            
            if self.writer:
                self.writer.release()
                self.writer = None
        
        if self.writer is None:
            print(f"Warning: could not open VideoWriter for {self.output_path}")
    
    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for video encoding."""
        src = frame
        if src.dtype != np.uint8:
            src = (np.clip(src, 0.0, 1.0) * 255).astype(np.uint8)
        if src.ndim == 2:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        return src
    
    def write_frame_with_hud(self, frame: np.ndarray, game: Any):
        """Write frame with HUD overlay showing scores."""
        if self.writer is None:
            self.start(frame)
        
        if self.writer is None:
            return
        
        try:
            src = self._prepare_frame(frame)
            
            # Calculate scores
            coop_score = 0
            scores = {}
            try:
                for agent_id, obj in game.gameObjects.items():
                    if agent_id.startswith('ai_rl_') and obj is not None:
                        score = int(getattr(obj, 'score', 0) or 0)
                        coop_score += score
                        scores[agent_id] = score
            except Exception:
                pass
            
            # Create HUD
            hud_lines = [f"Coop: {coop_score}"]
            for label in ('ai_rl_1', 'ai_rl_2'):
                hud_lines.append(f"{label}: {scores.get(label, 0)}")
            
            # Add HUD to frame (this preserves the original game dimensions)
            frame_with_hud = self._add_hud_to_frame(src, hud_lines)
            
            # Never resize - the frame should already be the correct size
            self.writer.write(frame_with_hud)
            
        except Exception as e:
            print(f"Warning: failed to write video frame: {e}")
    
    def _add_hud_to_frame(self, frame: np.ndarray, hud_lines: List[str]) -> np.ndarray:
        """Add HUD overlay to frame with pure black background."""
        h, w = frame.shape[:2]
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        line_height = cv2.getTextSize('Tg', font, scale, thickness)[0][1] + 6
        
        # Use consistent HUD height calculated during initialization
        if self.hud_height is None:
            pad_height = max(48, 12 + len(hud_lines) * line_height)
        else:
            pad_height = self.hud_height
        
        # Create canvas with HUD area - preserve original game frame exactly
        canvas = np.zeros((h + pad_height, w, 3), dtype=np.uint8)
        canvas[0:h, 0:w] = frame
        
        # Draw pure black background for HUD area
        cv2.rectangle(canvas, (0, h), (w, h + pad_height), (0, 0, 0), -1)
        
        # Draw HUD text
        start_y = h + 12 + line_height
        for i, text in enumerate(hud_lines):
            y = start_y + i * line_height
            # Shadow for better readability
            cv2.putText(canvas, text, (9, y+1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            # White text
            cv2.putText(canvas, text, (8, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return canvas
    
    def stop(self):
        """Stop video recording and release resources."""
        if self.writer:
            self.writer.release()
            self.writer = None