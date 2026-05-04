import os
import time
import logging
import threading

log = logging.getLogger(__name__)

GPIO_LOCK_PIN   = int(os.getenv("GPIO_LOCK_PIN", "17"))
UNLOCK_DURATION = float(os.getenv("UNLOCK_DURATION", "3.0"))


class DoorLock:
    """Controls a GPIO-connected door lock relay.

    Wiring convention: HIGH = locked, LOW = unlocked (active-low relay).
    Gracefully degrades to simulation mode when RPi.GPIO is unavailable,
    so the same code runs on Raspberry Pi and on x86 development machines.
    """

    def __init__(self):
        self._thread_lock = threading.Lock()
        self._is_unlocked = False
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(GPIO_LOCK_PIN, GPIO.OUT, initial=GPIO.HIGH)
            self._gpio = GPIO
            self._sim = False
            log.info("GPIO door lock initialised on BCM pin %d", GPIO_LOCK_PIN)
        except (ImportError, RuntimeError) as exc:
            self._gpio = None
            self._sim = True
            log.info("GPIO unavailable (%s) — running in simulation mode", exc)

    def unlock(self, duration: float | None = None) -> None:
        dur = duration if duration is not None else UNLOCK_DURATION
        # Run in a daemon thread so the camera loop is never blocked
        threading.Thread(target=self._unlock_cycle, args=(dur,), daemon=True).start()

    def _unlock_cycle(self, duration: float) -> None:
        with self._thread_lock:
            if self._is_unlocked:
                return  # let the running timer finish rather than resetting it
            self._is_unlocked = True

            if self._sim:
                log.info("[SIM] Door UNLOCKED for %.1f s", duration)
            else:
                self._gpio.output(GPIO_LOCK_PIN, self._gpio.LOW)
                log.info("Door UNLOCKED for %.1f s", duration)

            time.sleep(duration)
            self._is_unlocked = False

            if self._sim:
                log.info("[SIM] Door LOCKED")
            else:
                self._gpio.output(GPIO_LOCK_PIN, self._gpio.HIGH)
                log.info("Door LOCKED")

    def cleanup(self) -> None:
        if not self._sim and self._gpio:
            self._gpio.cleanup()
            log.info("GPIO cleaned up")
