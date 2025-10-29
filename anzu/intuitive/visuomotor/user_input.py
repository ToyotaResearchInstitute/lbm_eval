"""
Provides simple wrappers for accessing specific joysticks / gamepads via
PyGame.

Joysticks currently supported:

  - Logitech Gamepad F310 (suggested)
    https://www.amazon.com/Logitech-940-000110-Gamepad-F310/dp/B003VAHYQY
  - Logitech Gamepad F710
    https://www.amazon.com/Logitech-940-000117-Gamepad-F710/dp/B0041RR0TW
  - Xbox One S Controller
    https://www.amazon.com/Microsoft-Xbox-Wireless-Controller-Renewed/dp/B01N06M83E
  - Logitech Dual Action
    (old, no purchase link)
  - 8Bitdo SN30 Pro
    https://www.amazon.com/dp/B0748S1VDC
  - TigerGame PS/PS2 adapter (also sold as Super Dual Box Pro, among other names)
    (old, no purchase link)

Inspired by drake setup:
https://github.com/RobotLocomotion/drake/blob/v0.26.0/examples/manipulation_station/end_effector_teleop_dualshock4.py  # noqa

Caution: Double-check button mappings when adding a new controller. To check
mappings, use pygame demo:
- https://www.pygame.org/docs/ref/joystick.html
- https://github.com/pygame/pygame/blob/2.0.1/docs/reST/ref/code_examples/joystick_calls.py
"""  # noqa

import atexit
import dataclasses as dc
from enum import Enum
import os
from threading import Lock, Thread
import time
import weakref

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # noqa

# To map from L2/R2 trigger axes to a button boolean.
TRIGGER_TO_BUTTON_THRESHOLD = 0.5


def pygame():
    """Returns 'import pygame', deferring module loading until the first time
    its needed so that rollouts can avoid a dependency on teleop input devices.
    """
    import pygame as _pygame

    return _pygame


class JoystickButton(Enum):
    # N.B. These buttons are defined in (A, B, X, Y) order, according to
    # Logitech controller.
    # This contrasts w/ PlayStation controller, which is (X, O, ☐, Δ).
    A_BUTTON = 0
    B_BUTTON = 1
    X_BUTTON = 2
    Y_BUTTON = 3
    L1_BUTTON = 4
    R1_BUTTON = 5
    SELECT_BUTTON = 6
    START_BUTTON = 7
    LEFTJOY_PRESSED = 8
    RIGHTJOY_PRESSED = 9
    # These can be either buttons or axis depending on the controller, thus
    # the overlap with JoystickAxis.
    L2_BUTTON = 10
    R2_BUTTON = 11


class JoystickAxis(Enum):
    LEFTJOY_UP_DOWN = 0  # Up: -1, Down: 1
    LEFTJOY_LEFT_RIGHT = 1  # Left: -1, Right: 1
    RIGHTJOY_LEFT_RIGHT = 2  # Left: -1, Right: 1
    RIGHTJOY_UP_DOWN = 3  # Up: -1, Down: 1
    L2_BUTTON = 4  # Release: -1, Press: 1
    R2_BUTTON = 5  # Release: -1, Press: 1

    @classmethod
    def get_triggers(cls):
        return [
            cls.L2_BUTTON,
            cls.R2_BUTTON,
        ]


class JoystickHat(Enum):
    # (hat index, hat sub-index)
    LEFT_RIGHT = (0, 0)
    UP_DOWN = (0, 1)


@dc.dataclass
class _JoystickMapping:
    buttons: dict
    axes: dict
    event_buttons: dict


def _get_joystick_mapping(controller_name):
    """
    This mapps from JoystickButton to button index in pygame for different
    controllers.
    """
    if controller_name in ["Logitech Gamepad F710", "Logitech Gamepad F310"]:
        return _JoystickMapping(
            buttons={
                JoystickButton.A_BUTTON: 0,
                JoystickButton.B_BUTTON: 1,
                JoystickButton.X_BUTTON: 2,
                JoystickButton.Y_BUTTON: 3,
                JoystickButton.L1_BUTTON: 4,
                JoystickButton.R1_BUTTON: 5,
                JoystickButton.SELECT_BUTTON: 6,
                JoystickButton.START_BUTTON: 7,
                JoystickButton.LEFTJOY_PRESSED: 9,
                JoystickButton.RIGHTJOY_PRESSED: 10,
                # These are axes, not buttons.
                JoystickButton.L2_BUTTON: None,
                JoystickButton.R2_BUTTON: None,
            },
            axes={
                JoystickAxis.LEFTJOY_UP_DOWN: 1,
                JoystickAxis.LEFTJOY_LEFT_RIGHT: 0,
                JoystickAxis.RIGHTJOY_LEFT_RIGHT: 3,
                JoystickAxis.RIGHTJOY_UP_DOWN: 4,
                JoystickAxis.L2_BUTTON: 2,
                JoystickAxis.R2_BUTTON: 5,
            },
            event_buttons={},
        )
    elif controller_name == "Logitech Dual Action":
        return _JoystickMapping(
            buttons={
                JoystickButton.A_BUTTON: 1,
                JoystickButton.B_BUTTON: 2,
                JoystickButton.X_BUTTON: 0,
                JoystickButton.Y_BUTTON: 3,
                JoystickButton.L1_BUTTON: 4,
                JoystickButton.R1_BUTTON: 5,
                JoystickButton.SELECT_BUTTON: 8,
                JoystickButton.START_BUTTON: 9,
                JoystickButton.LEFTJOY_PRESSED: 10,
                JoystickButton.RIGHTJOY_PRESSED: 11,
                JoystickButton.L2_BUTTON: 6,
                JoystickButton.R2_BUTTON: 7,
            },
            axes={
                JoystickAxis.LEFTJOY_UP_DOWN: 1,
                JoystickAxis.LEFTJOY_LEFT_RIGHT: 0,
                JoystickAxis.RIGHTJOY_LEFT_RIGHT: 2,
                JoystickAxis.RIGHTJOY_UP_DOWN: 3,
                # These are buttons, not axes.
                JoystickAxis.L2_BUTTON: None,
                JoystickAxis.R2_BUTTON: None,
            },
            event_buttons={},
        )
    elif controller_name == "Xbox One S Controller":
        return _JoystickMapping(
            buttons={
                JoystickButton.A_BUTTON: 0,
                JoystickButton.B_BUTTON: 1,
                JoystickButton.X_BUTTON: 2,
                JoystickButton.Y_BUTTON: 3,
                JoystickButton.L1_BUTTON: 4,
                JoystickButton.R1_BUTTON: 5,
                JoystickButton.SELECT_BUTTON: 6,
                JoystickButton.START_BUTTON: 7,
                JoystickButton.LEFTJOY_PRESSED: 9,
                JoystickButton.RIGHTJOY_PRESSED: 10,
                # These are axes, not buttons.
                JoystickButton.L2_BUTTON: None,
                JoystickButton.R2_BUTTON: None,
            },
            axes={
                JoystickAxis.LEFTJOY_UP_DOWN: 1,
                JoystickAxis.LEFTJOY_LEFT_RIGHT: 0,
                JoystickAxis.RIGHTJOY_LEFT_RIGHT: 3,
                JoystickAxis.RIGHTJOY_UP_DOWN: 4,
                JoystickAxis.L2_BUTTON: 2,
                JoystickAxis.R2_BUTTON: 5,
            },
            event_buttons={},
        )
    elif (
        controller_name
        == "WiseGroup.,Ltd TigerGame PS/PS2 Game Controller Adapter"
    ):  # noqa
        return _JoystickMapping(
            buttons={
                JoystickButton.A_BUTTON: 2,
                JoystickButton.B_BUTTON: 1,
                JoystickButton.X_BUTTON: 3,
                JoystickButton.Y_BUTTON: 0,
                JoystickButton.L1_BUTTON: 6,
                JoystickButton.L2_BUTTON: 4,
                JoystickButton.R1_BUTTON: 7,
                JoystickButton.R2_BUTTON: 5,
                JoystickButton.SELECT_BUTTON: 9,
                JoystickButton.START_BUTTON: 8,
                JoystickButton.LEFTJOY_PRESSED: 10,
                JoystickButton.RIGHTJOY_PRESSED: 11,
            },
            axes={
                JoystickAxis.LEFTJOY_UP_DOWN: 1,
                JoystickAxis.LEFTJOY_LEFT_RIGHT: 0,
                JoystickAxis.RIGHTJOY_LEFT_RIGHT: 2,
                JoystickAxis.RIGHTJOY_UP_DOWN: 3,
                # These are buttons, not axes.
                JoystickAxis.L2_BUTTON: None,
                JoystickAxis.R2_BUTTON: None,
            },
            event_buttons={
                (JoystickHat.LEFT_RIGHT, -1): 15,
                (JoystickHat.LEFT_RIGHT, 1): 13,
                (JoystickHat.UP_DOWN, -1): 14,
                (JoystickHat.UP_DOWN, 1): 12,
            },
        )
    elif controller_name == "Sony Computer Entertainment Wireless Controller":
        # This represents (at least) the 8Bitdo SN30 Pro.
        # TODO(eric.cousineau): Which input mode is this? XInput, DInput,
        # Switch, or macOS?
        return _JoystickMapping(
            buttons={
                JoystickButton.A_BUTTON: 1,
                JoystickButton.B_BUTTON: 0,
                JoystickButton.X_BUTTON: 2,
                JoystickButton.Y_BUTTON: 3,
                JoystickButton.L1_BUTTON: 4,
                JoystickButton.L2_BUTTON: 6,
                JoystickButton.R1_BUTTON: 5,
                JoystickButton.R2_BUTTON: 7,
                JoystickButton.SELECT_BUTTON: 8,
                JoystickButton.START_BUTTON: 9,
                JoystickButton.LEFTJOY_PRESSED: 11,
                JoystickButton.RIGHTJOY_PRESSED: 12,
            },
            axes={
                JoystickAxis.LEFTJOY_UP_DOWN: 1,
                JoystickAxis.LEFTJOY_LEFT_RIGHT: 0,
                JoystickAxis.RIGHTJOY_LEFT_RIGHT: 3,
                JoystickAxis.RIGHTJOY_UP_DOWN: 4,
                # These are axes, but 1 if pressed, -1 otherwise (which seems
                # odd).
                JoystickAxis.L2_BUTTON: None,
                JoystickAxis.R2_BUTTON: None,
            },
            event_buttons={},
        )
    elif controller_name == "Sony PLAYSTATION(R)3 Controller":
        # Even with support for simulating the hat, we still can't
        # support this controller.  When pygame is built against
        # libSDL-1.2 (like the system version on Ubuntu 20.04), the
        # library opens /dev/input/js0 (using the joystick device
        # API), and the controller shows up with 17 buttons, including
        # the hat, which would work.  However, our pygame inside anzu
        # is built against libSDL-2, where the library opens
        # /dev/input/event* (using the event API) and the controller
        # appears with only 13 buttons (missing the D-pad) and still
        # no hats.
        raise RuntimeError(
            f"DO NOT USE. Joystick does not seem to have 'hat' buttons "
            f"available via pygame.joystick: {controller_name}."
        )
    else:
        raise RuntimeError(f"Joystick not yet mapped: {controller_name}")


def get_pygame_joysticks_with_anzu_mappings():
    """
    This goes through all the joystick devices (via pygame) and to enumerate
    which ones we already have mappings for.

    This also initialiizes the minimum set of pygame modules.
    """
    # This change is introduced because spacemouse is also registered
    # as a joystick device by pygame, and that makes selecting based
    # on joystick index a bit more tricky.
    pygame().display.init()  # Necessary to use `pygame.event`.
    pygame().joystick.init()
    valid_joysticks = []
    for idx in range(pygame().joystick.get_count()):
        joystick = pygame().joystick.Joystick(idx)
        try:
            # Check if we know how to work with this device.
            _ = _get_joystick_mapping(joystick.get_name())
            valid_joysticks.append(joystick)
        except RuntimeError:
            print(f"Unmapped device for joystick: {joystick.get_name()}")
    return valid_joysticks


class JoystickWrapper:
    _SINGLETON = None

    def __init__(self, joystick):
        self._mapping = _get_joystick_mapping(joystick.get_name())
        self._joystick = joystick
        self._lock = Lock()
        self._latest_events = self._get_events()

        # N.B. Not using lock on this variable seems to *improve* thread
        # synchronization issues.
        self._running = True
        self._thread = Thread(target=self._thread_loop)
        self._thread.daemon = True
        self._thread.start()

        # Use weak reference so that closure doesn't artificially keep this
        # object alive.
        atexit.register(JoystickWrapper._stop_if_alive, weakref.ref(self))

    @staticmethod
    def _stop_if_alive(weak_self):
        self = weak_self()
        if self is not None:
            self._running = False

    @classmethod
    def make_singleton(cls, *, throw_on_error=True):
        wrapper = cls._SINGLETON
        if wrapper is None:
            valid_joysticks = get_pygame_joysticks_with_anzu_mappings()
            if throw_on_error:
                # Check if we only have one valid joystick plugged in.
                assert len(valid_joysticks) == 1, valid_joysticks
            else:
                if len(valid_joysticks) != 1:
                    return None
            joystick = valid_joysticks[0]
            joystick.init()
            wrapper = JoystickWrapper(joystick)
            cls._SINGLETON = wrapper
        return wrapper

    @classmethod
    def make_dummy_test_singleton(cls):
        """
        Call this in a unittest to stub out required joystick functionality.

        Please note that this is not a true mock object, so you may need
        to consider alternatives if you need non-trivial functionality (e.g.
        events).
        """
        assert cls._SINGLETON is None
        cls._SINGLETON = (
            "<fake object created by JoystickWrapper."
            "make_dummy_test_singleton()>"
        )

    def can_reset(self):
        events = self.get_events()
        return not self.are_any_buttons_pressed(events)

    def reset(self):
        # N.B. Should ensure statefulness is only in polling class, not in
        # pygame device... somehow?
        pass

    def _thread_loop(self):
        print(
            f"Starting thread loop for pygame "
            f"Joystick {self._joystick.get_id()}"
        )
        dt = 0.001
        while self._running:
            events = self._get_events()
            with self._lock:
                self._latest_events = events
            time.sleep(dt)

    def get_events(self):
        with self._lock:
            events = self._latest_events
        return events

    def _get_events(self):
        """
        Polls event directly from pygame device.

        (So we can have multiple subscribers to same joystick)
        """
        # TODO(eric): Is it a bad thing to pump events here?
        pygame().event.pump()

        events = dict()

        def deadband(x):
            # Some controllers seem to have some stiction.
            if abs(x) < 0.12:
                return 0.0
            else:
                return x

        # To get mappings, use example code here:
        # https://www.pygame.org/docs/ref/joystick.html#controller-mappings
        # - Axes.
        for enum in JoystickAxis:
            axis_idx = self._mapping.axes[enum]
            if axis_idx is not None:
                events[enum] = deadband(self._joystick.get_axis(axis_idx))
            else:
                events[enum] = -1.0

        # - Hat (direction pad).
        for enum in JoystickHat:
            hat_index, hat_subindex = enum.value
            if self._joystick.get_numhats() > 0:
                hat_data = self._joystick.get_hat(hat_index)
            else:
                hat_data = (0, 0)
            events[enum] = hat_data[hat_subindex]

        # Buttons.
        for enum in JoystickButton:
            button_idx = self._mapping.buttons[enum]
            if button_idx is not None:
                events[enum] = self._joystick.get_button(button_idx)

        # Some controllers implement features using buttons instead of
        # other options (like hats).  Allow those controllers to
        # simulate arbitrary events with buttons.
        for (event, value), button in self._mapping.event_buttons.items():
            if self._joystick.get_button(button):
                events[event] = value

        # For whatever reason, the L2 and R2 axis start at zero, even though
        # they should start at -1.0. To fix, we latch read values of 0 to -1.0
        # at the very start.

        # Map the (fixed) trigger values to booleans according to threshold.
        triggers = [JoystickAxis.L2_BUTTON, JoystickAxis.R2_BUTTON]
        trigger_to_button = {
            JoystickAxis.L2_BUTTON: JoystickButton.L2_BUTTON,
            JoystickAxis.R2_BUTTON: JoystickButton.R2_BUTTON,
        }
        for trigger in triggers:
            if self._mapping.axes[trigger] is not None:
                events[trigger_to_button[trigger]] = (
                    events[trigger] >= TRIGGER_TO_BUTTON_THRESHOLD
                )

        return events

    @staticmethod
    def _make_empty():
        empty = dict()
        empty[JoystickAxis.LEFTJOY_LEFT_RIGHT] = 0
        empty[JoystickAxis.LEFTJOY_UP_DOWN] = 0
        empty[JoystickAxis.RIGHTJOY_LEFT_RIGHT] = 0
        empty[JoystickAxis.RIGHTJOY_UP_DOWN] = 0
        empty[JoystickAxis.L2_BUTTON] = -1.0
        empty[JoystickAxis.R2_BUTTON] = -1.0
        empty[JoystickHat.LEFT_RIGHT] = 0
        empty[JoystickHat.UP_DOWN] = 0
        empty[JoystickButton.A_BUTTON] = False
        empty[JoystickButton.B_BUTTON] = False
        empty[JoystickButton.X_BUTTON] = False
        empty[JoystickButton.Y_BUTTON] = False
        empty[JoystickButton.L1_BUTTON] = False
        empty[JoystickButton.R1_BUTTON] = False
        empty[JoystickButton.L2_BUTTON] = False
        empty[JoystickButton.R2_BUTTON] = False
        empty[JoystickButton.SELECT_BUTTON] = False
        empty[JoystickButton.START_BUTTON] = False
        empty[JoystickButton.LEFTJOY_PRESSED] = False
        empty[JoystickButton.RIGHTJOY_PRESSED] = False
        return empty

    @classmethod
    def are_any_buttons_pressed(cls, events):
        empty = cls._make_empty()
        assert events.keys() == empty.keys(), set(events.keys()) ^ set(
            empty.keys()
        )
        return events != empty


DEFAULT_DT = 0.005


def wait_for_press(joystick, buttons, *, dt=DEFAULT_DT):
    # Wait for press.
    while True:
        events = joystick.get_events()
        for button in buttons:
            if events[button]:
                return button
        time.sleep(dt)


def wait_for_release(joystick, button, *, dt=DEFAULT_DT):
    events = joystick.get_events()
    while events[button]:
        events = joystick.get_events()
        time.sleep(dt)


def wait_for_press_and_release(joystick, buttons, *, dt=DEFAULT_DT):
    button = wait_for_press(joystick, buttons, dt=dt)
    wait_for_release(joystick, button, dt=dt)
    return button


def check_for_press_and_release(joystick, button, *, dt=DEFAULT_DT):
    events = joystick.get_events()
    if events[button]:
        wait_for_release(joystick, button, dt=dt)
        return True
    return False
