"""
core/gpio_controller.py
────────────────────────
Thin wrapper around Jetson.GPIO for controlling a 4-channel relay board
(SRD-05VDC-SL-C, active-LOW) connected to the Jetson Orin Nano 40-pin header.

Pin mapping (BOARD numbering)
─────────────────────────────
  IN1 → Pin 16  — Application ready / model loaded
  IN2 → Pin 15  — Detection / camera stream active
  IN3 → Pin 13  — Detection / camera stream active (paired with IN2)
  IN4 → Pin 11  — Counting complete

Relay polarity
──────────────
  The SRD-05VDC-SL-C energises the coil when its IN line is driven LOW.
  Therefore:
    • Relay ON  (indicator lit)  → GPIO.output(pin, GPIO.LOW)
    • Relay OFF (indicator off)  → GPIO.output(pin, GPIO.HIGH)

Fallback
────────
  If Jetson.GPIO is not available (e.g., running on a development PC),
  a MockGPIO class is used automatically so the application still runs
  without any GPIO hardware present.
"""

import logging
from typing import Optional

from core.config import (
    GPIO_PIN_READY,
    GPIO_PIN_DETECT_A,
    GPIO_PIN_DETECT_B,
    GPIO_PIN_COMPLETE,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# GPIO backend — real Jetson.GPIO or mock fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    import Jetson.GPIO as _GPIO
    _gpio_available = True
    logger.info("GPIOController: Jetson.GPIO loaded successfully.")
except ImportError:
    _gpio_available = False
    logger.warning(
        "GPIOController: Jetson.GPIO not found — running with mock GPIO. "
        "Relay indicators will be simulated (log-only)."
    )

    class _MockGPIO:
        """Drop-in mock that logs GPIO actions instead of driving hardware."""
        BOARD = "BOARD"
        OUT   = "OUT"
        HIGH  = 1
        LOW   = 0

        def setmode(self, mode):
            logger.debug(f"[MockGPIO] setmode({mode})")

        def setup(self, pin, direction, initial=None):
            init_str = f", initial={initial}" if initial is not None else ""
            logger.debug(f"[MockGPIO] setup(pin={pin}, direction={direction}{init_str})")

        def output(self, pin, value):
            level = "LOW (ON)" if value == self.LOW else "HIGH (OFF)"
            logger.debug(f"[MockGPIO] output(pin={pin}, {level})")

        def cleanup(self):
            logger.debug("[MockGPIO] cleanup()")

    _GPIO = _MockGPIO()


# ─────────────────────────────────────────────────────────────────────────────
# GPIOController
# ─────────────────────────────────────────────────────────────────────────────
class GPIOController:
    """
    Manages the 4-channel relay board indicators for the fish detection app.

    All methods are safe to call from the Qt main thread — GPIO writes are
    fast (microseconds) and non-blocking.

    Usage
    ─────
        gpio = GPIOController()
        gpio.set_ready()          # app started, model loaded
        gpio.set_detecting(True)  # detection or camera stream started
        gpio.set_detecting(False) # detection or camera stream ended
        gpio.set_complete()       # counting finished
        gpio.reset_all()          # turn everything off
        gpio.cleanup()            # call once on application exit
    """

    _ALL_PINS = (
        GPIO_PIN_READY,
        GPIO_PIN_DETECT_A,
        GPIO_PIN_DETECT_B,
        GPIO_PIN_COMPLETE,
    )

    def __init__(self):
        self._initialised = False
        try:
            _GPIO.setmode(_GPIO.BOARD)
            for pin in self._ALL_PINS:
                # Initialise all relays OFF (HIGH = relay coil de-energised)
                _GPIO.setup(pin, _GPIO.OUT, initial=_GPIO.HIGH)
            self._initialised = True
            logger.info(
                f"GPIOController: pins {self._ALL_PINS} initialised as OUTPUT (HIGH/OFF)."
            )
        except Exception as exc:
            logger.error(f"GPIOController: init failed — {exc}")

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _write(self, pin: int, value: int):
        """Drive a single pin; logs and swallows any hardware error."""
        try:
            _GPIO.output(pin, value)
            level = "LOW (ON)" if value == _GPIO.LOW else "HIGH (OFF)"
            logger.info(f"GPIOController: pin {pin} → {level}")
        except Exception as exc:
            logger.error(f"GPIOController: failed to write pin {pin} — {exc}")

    def _on(self, pin: int):
        """Energise relay (active-LOW → drive pin LOW)."""
        self._write(pin, _GPIO.LOW)

    def _off(self, pin: int):
        """De-energise relay (drive pin HIGH)."""
        self._write(pin, _GPIO.HIGH)

    # ── Public API ────────────────────────────────────────────────────────────
    def set_ready(self):
        """
        Application is ready for inference (model loaded).
        Lights up IN1 (pin 16). Ensures detect and complete indicators are off.
        """
        self._off(GPIO_PIN_DETECT_A)
        self._off(GPIO_PIN_DETECT_B)
        self._off(GPIO_PIN_COMPLETE)
        self._on(GPIO_PIN_READY)
        logger.info("GPIOController: state → READY")

    def set_detecting(self, active: bool):
        """
        Signal that detection / camera stream has started or stopped.

        active=True  → lights up IN2 (pin 15) and IN3 (pin 13)
        active=False → turns off IN2 and IN3; leaves READY and COMPLETE untouched
        """
        if active:
            self._on(GPIO_PIN_DETECT_A)
            self._on(GPIO_PIN_DETECT_B)
            logger.info("GPIOController: state → DETECTING")
        else:
            self._off(GPIO_PIN_DETECT_A)
            self._off(GPIO_PIN_DETECT_B)
            logger.info("GPIOController: state → DETECTING off")

    def set_complete(self):
        """
        Counting is complete.
        Turns off ALL other indicators and lights up only IN4 (pin 11).
        State machine: pin 16 (start) → pins 15+13 (detecting) → pin 11 (done).
        """
        self._off(GPIO_PIN_READY)
        self._off(GPIO_PIN_DETECT_A)
        self._off(GPIO_PIN_DETECT_B)
        self._on(GPIO_PIN_COMPLETE)
        logger.info("GPIOController: state → COMPLETE")

    def reset_all(self):
        """Turn all relay indicators off (all pins HIGH)."""
        for pin in self._ALL_PINS:
            self._off(pin)
        logger.info("GPIOController: all relays reset (OFF)")

    def cleanup(self):
        """
        Release all GPIO resources. Call once when the application exits.
        Safe to call even if initialisation failed.
        """
        try:
            self.reset_all()
            _GPIO.cleanup()
            logger.info("GPIOController: GPIO cleanup complete.")
        except Exception as exc:
            logger.error(f"GPIOController: cleanup error — {exc}")
