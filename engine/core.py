import time
from typing import Dict, Optional


class Engine:
    def __init__(self, game,
                 tick_rate: int = 20,
                 run_max_speed: bool = False,
                 run_fixed_delta: bool = False,
                 ):
        self.game = game
        self.tick_interval = 1.0 / tick_rate
        self.running = False
        self.tick_count = 0
        self.start_time = time.time()
        self.last_time = time.time()
        self.run_max_speed = run_max_speed
        self.run_fixed_delta = run_fixed_delta
        self.intent_buffer: Dict[str, str] = {}  # agent_id -> action

    def submit_intent(self, agent_id: str, action: str):
        self.intent_buffer[agent_id] = action

    def tick(self, delta_time: Optional[float] = None):
        self.game.step(self.intent_buffer, delta_time)
        self.intent_buffer.clear()
        self.tick_count += 1

    def start_loop(self, max_ticks: Optional[int] = None):
        self.running = True
        self.start_time = time.time()
        self.last_time = self.start_time
        while self.running and not self.game.done:
            if max_ticks is not None and self.tick_count >= max_ticks:
                break

            start_time = time.time()
            delta_time = start_time - self.last_time
            self.last_time = start_time
            if self.run_fixed_delta:
                delta_time = self.tick_interval

            self.tick(delta_time)
            elapsed = time.time() - start_time
            if not self.run_max_speed:
                sleep_time = max(0, self.tick_interval - elapsed)
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
