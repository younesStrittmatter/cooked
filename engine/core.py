import time
from typing import Dict, Optional


class Engine:
    def __init__(self,
                 game,
                 tick_rate: int,
                 is_max_speed: bool,
                 ):
        self.game = game
        self.tick_interval = 1.0 / tick_rate
        self.running = False
        self.tick_count = 0

        if not is_max_speed:
            self.start_time = time.time()
            self.last_time = self.start_time
        else:
            self.start_time = 0
            self.last_time = 0

        self.is_max_speed = is_max_speed

        self.intent_buffer: Dict[str, str] = {}

    def submit_intent(self, agent_id: str, action: str):
        # Don't overwrite existing intents with None - None means "no new action, keep existing"
        if action is None:
            return
            
        # Log when an existing intent for the agent is being overwritten.
        prev = self.intent_buffer.get(agent_id)
        if prev is not None and prev != action:
            try:
                print(f"Engine: overwriting intent for {agent_id} at tick={getattr(self, 'tick_count', None)} prev={prev} new={action}")
            except Exception:
                pass
        self.intent_buffer[agent_id] = action

    def tick(self, delta_time: Optional[float] = None):
        self.game.step(self.intent_buffer, delta_time)
        self.intent_buffer.clear()
        self.tick_count += 1

    def start_loop(self, max_ticks: Optional[int] = None):
        self.running = True
        if not self.is_max_speed:
            self.start_time = time.time()
            self.last_time = self.start_time
        else:
            self.start_time = 0
            self.last_time = 0

        while self.running and not self.game.done:
            if max_ticks is not None and self.tick_count >= max_ticks:
                break

            if not self.is_max_speed:
                start_time = time.time()
                delta_time = start_time - self.last_time
                self.last_time = start_time
            else:
                delta_time = self.tick_interval

            self.tick(delta_time)
            if not self.is_max_speed:
                elapsed = time.time() - start_time
                sleep_time = max(0, self.tick_interval - elapsed)
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
