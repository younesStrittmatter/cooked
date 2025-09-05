import time
import threading
from typing import Dict, Optional

class Engine:
    def __init__(self,
                 game,
                 tick_rate: int,
                 is_max_speed: bool,
                 replay_recorder=None):
        self.game = game
        self.tick_interval = 1.0 / tick_rate
        self.running = False
        self.tick_count = 0
        self.is_max_speed = is_max_speed

        # Use monotonic clock for scheduling
        self._mono = time.monotonic

        self.on_tick_callbacks = []
        self.on_game_end_callbacks = []
        self.recorder = replay_recorder

        # Intent buffer + tiny lock for safety across greenlets/threads
        self._intent_lock = threading.Lock()
        self.intent_buffer: Dict[str, str] = {}

    def add_on_game_end_callback(self, fn):
        self.on_game_end_callbacks.append(fn)

    def add_on_tick_callback(self, fn):
        self.on_tick_callbacks.append(fn)

    def submit_intent(self, agent_id: str, action: str):
        # Safe insert; cheap critical section
        with self._intent_lock:
            if isinstance(action, str):
                action = {"type": action}
            elif action is None:
                action = {}
            self.intent_buffer[agent_id] = action


    def _drain_intents(self) -> Dict[str, str]:
        # Swap buffer to avoid holding the lock while stepping
        with self._intent_lock:
            intents = self.intent_buffer
            self.intent_buffer = {}
        return intents

    def tick(self, delta_time: Optional[float] = None):
        intents = self._drain_intents()
        # Log intents deterministically at this tick
        if self.recorder and intents:
            for agent_id, intent in intents.items():
                self.recorder.log_intent(self.tick_count, agent_id, intent)

        # Advance game one fixed step
        self.game.step(intents, delta_time)

        self.tick_count += 1

        # Callbacks should be lightweight (mark-only for emit, etc.)
        for callback in self.on_tick_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"[Engine] Tick callback failed: {e}")

    def start_loop(self, max_ticks: Optional[int] = None):
        """Fixed-timestep loop with exact scheduling and catch-up."""
        self.running = True
        dt = self.tick_interval
        mono = self._mono

        # Next deadline in monotonic time
        next_deadline = mono()

        try:
            while self.running and not self.game.done:
                now = mono()

                # Catch-up: run as many ticks as deadlines we’ve passed
                # Clamp catch-up to avoid spiral-of-death (e.g., 0.25s worth)
                catchup_limit = 0.25
                late = now - next_deadline
                if late > catchup_limit:
                    # Skip ahead to avoid doing hundreds of back-to-back ticks
                    skipped = int(late // dt) - int(catchup_limit // dt)
                    next_deadline += skipped * dt

                while now >= next_deadline and not self.game.done:
                    self.tick(dt)
                    next_deadline += dt

                    if max_ticks is not None and self.tick_count >= max_ticks:
                        self.running = False
                        break

                    now = mono()

                # Sleep until next deadline (cooperative under Eventlet)
                sleep_time = max(0.0, next_deadline - now)
                # Avoid very tiny sleeps that lead to spin; still yields cooperatively
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We're late; yield minimally
                    time.sleep(0)
        finally:
            for cb in self.on_game_end_callbacks:
                try:
                    cb()
                except Exception as e:
                    print(f"[Engine] Game end callback failed: {e}")

    def stop(self):
        self.running = False
