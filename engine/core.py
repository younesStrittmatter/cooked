import time
from typing import Dict, Optional


class Engine:
    def __init__(self,
                 game,
                 tick_rate: int,
                 is_max_speed: bool,
                 replay_recorder=None
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
        self.on_tick_callbacks = []
        self.on_game_end_callbacks = []
        self.recorder = replay_recorder

        self.intent_buffer: Dict[str, str] = {}

    def add_on_game_end_callback(self, fn):
        self.on_game_end_callbacks.append(fn)

    def add_on_tick_callback(self, fn):
        self.on_tick_callbacks.append(fn)

    def submit_intent(self, agent_id: str, action: str):
        self.intent_buffer[agent_id] = action

    def tick(self, delta_time: Optional[float] = None):
        for agent_id, intent in self.intent_buffer.items():
            self.recorder.log_intent(self.tick_count, agent_id, intent)
        self.game.step(self.intent_buffer, delta_time)
        self.intent_buffer.clear()
        self.tick_count += 1

        for callback in self.on_tick_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"[Engine] Tick callback failed: {e}")

    def start_loop(self, max_ticks: Optional[int] = None):
        self.running = True
        tick_interval = self.tick_interval
        self.accumulated_lag = 0.0

        if not self.is_max_speed:
            self.last_time = time.time()

        while self.running and not self.game.done:
            if max_ticks is not None and self.tick_count >= max_ticks:
                break

            if self.is_max_speed:
                # Max speed: no timing, just tick as fast as possible
                self.tick(tick_interval)
            else:
                # Real-time mode: accumulate actual time
                current_time = time.time()
                frame_time = current_time - self.last_time
                self.last_time = current_time

                # Clamp to avoid spiral of death
                frame_time = min(frame_time, 0.25)
                self.accumulated_lag += frame_time

                # Run as many ticks as we’re behind
                while self.accumulated_lag >= tick_interval:
                    self.tick(tick_interval)
                    self.accumulated_lag -= tick_interval

                # Sleep to maintain pacing
                sleep_time = tick_interval - self.accumulated_lag
                if sleep_time > 0:
                    time.sleep(sleep_time)
        for cb in self.on_game_end_callbacks:
            try:
                cb()
            except Exception as e:
                print(f"[Engine] Game end callback failed: {e}")

    def stop(self):
        self.running = False
