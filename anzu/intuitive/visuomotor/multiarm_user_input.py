"""
Multiarm User Input classes and configurations. These classes allow the user
to handle pauses, resets, and signal successes and failures within the
`MultiarmLcmEnv` and `HardwareStationScenarioSimulationEnv` state machines.
"""
# TODO(eric.cousineau): Rename this to disambiguate from teleop input
# classes, or merge with teleop input classes.

import dataclasses as dc
import functools
import time

from anzu.common.user_input.footswitch import FootSwitch, Pedal
from anzu.intuitive.visuomotor.user_input import (
    JoystickButton,
    JoystickWrapper,
    check_for_press_and_release,
)


@dc.dataclass
class UserInputNoneConfig:
    def create(self):
        return UserInputNone()


@dc.dataclass
class UserInputFootSwitchConfig:
    def create(self):
        return UserInputFootSwitch()


@dc.dataclass
class UserInputJoystickConfig:
    def create(self):
        return UserInputJoystick()


class UserInputBase:
    def reset(self):
        raise NotImplementedError()

    def wait_for_start(self):
        self.print_start_signals()
        while not self.check_start_signals():
            time.sleep(1e-2)
        print("  Done")

    def print_success_signals(self):
        raise NotImplementedError()

    def check_success_signals(self):
        raise NotImplementedError()

    def print_start_signals(self):
        raise NotImplementedError()

    def check_start_signals(self):
        raise NotImplementedError()


class UserInputNone(UserInputBase):
    def reset(self):
        pass

    def wait_for_start(self):
        pass

    def print_success_signals(self):
        pass

    def check_success_signals(self):
        pass

    def print_start_signals(self):
        pass

    def check_start_signals(self):
        pass


class UserInputJoystick(UserInputBase):
    def __init__(self):
        self.joystick = JoystickWrapper.make_singleton()
        self.start_event = JoystickButton.X_BUTTON
        self.reset_event = JoystickButton.Y_BUTTON
        self.success_event = JoystickButton.SELECT_BUTTON
        self.failure_event = JoystickButton.START_BUTTON

    def reset(self):
        sleep_count = 0
        while not self.joystick.can_reset():
            sleep_count += 1
            if sleep_count == 10:
                print("Please release joystick buttons")
            time.sleep(0.05)
        self.joystick.reset()

    def print_success_signals(self):
        print(f"Events:")
        print(f"  {self.reset_event}: Reset/re-record demonstration")
        print(f"  {self.success_event}: Record success")
        print(f"  {self.failure_event}: Record failure")

    def check_success_signals(self):
        events = self.joystick.get_events()
        if events[self.reset_event]:
            return {"is_retry": True}
        elif events[self.success_event]:
            return {"is_success": True}
        elif events[self.failure_event]:
            return {"is_success": False}
        else:
            return None

    def print_start_signals(self):
        print(f"Press and release {self.start_event} to start")

    def check_start_signals(self):
        # TODO(eric): this currently won't return if user hold the button down
        # indefinitely. could block upstream unintentially.
        return check_for_press_and_release(self.joystick, self.start_event)


@functools.lru_cache
def _make_singleton_footswitch():
    return FootSwitch()


class UserInputFootSwitch(UserInputBase):
    def __init__(self):
        self.footswitch = _make_singleton_footswitch()
        self.start_event = Pedal.Left
        self.reset_event = Pedal.Left
        self.success_event = Pedal.Middle
        self.failure_event = Pedal.Right

    def reset(self):
        while True:
            events = self.footswitch.get_events()
            if self.footswitch.are_any_pressed(events):
                time.sleep(0.1)
            else:
                break

    def print_success_signals(self):
        print(f"Events:")
        print(f"  {self.reset_event}: Reset/re-record demonstration")
        print(f"  {self.success_event}: Record success")
        print(f"  {self.failure_event}: Record failure")

    def check_success_signals(self):
        events = self.footswitch.get_events()
        if events[self.reset_event]:
            return {"is_retry": True}
        elif events[self.success_event]:
            return {"is_success": True}
        elif events[self.failure_event]:
            return {"is_success": False}
        else:
            return None

    def print_start_signals(self):
        print(f"Press and release {self.start_event} to start")

    def check_start_signals(self):
        # TODO(eric): this currently won't return if user hold the button down
        # indefinitely. could block upstream unintentially.
        return check_for_press_and_release(self.footswitch, self.start_event)
