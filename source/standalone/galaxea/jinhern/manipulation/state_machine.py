"""Tick-based manipulation state machines.

Each state machine drives one skill through a ``ManipulatorInterface``. Call
``update()`` once per control tick (between calls the caller runs
``robot.step()``); it returns True when the skill has finished.

All waypoint offsets are constructor parameters. The defaults reproduce the
behavior of ``8_simpleAPI.py`` in the Isaac lift task — they are tuned for
that scene's object-position convention. Re-tune them (especially
``grasp_offset``) before running against real hardware.
"""

import math

import numpy as np

from .interface import LEFT, RIGHT, as_position, as_quaternion


class _ArmStateMachine:
    """Shared plumbing: state bookkeeping and gripper wait timing."""

    def __init__(self, robot, arm=LEFT, gripper_wait_s=1.0):
        self.robot = robot
        self.arm = arm
        self.gripper_wait_steps = max(1, math.ceil(gripper_wait_s / robot.step_dt))
        self.state = "IDLE"
        self.state_count = 0
        self.targets = {}

    @property
    def done(self):
        return self.state == "DONE"

    def _enter_state(self, state):
        self.state = state
        self.state_count = 0

    def _tick(self):
        self.state_count += 1


class PickStateMachine(_ArmStateMachine):
    """APPROACH above the object, DOWN to the grasp, CLOSE, LIFT.

    Waypoints are snapshotted once at ``start()`` so a moving object does not
    drag the targets around mid-grasp.
    """

    def __init__(
        self,
        robot,
        arm=LEFT,
        approach_offset=0.15,
        grasp_offset=0.07,
        lift_height=0.10,
        gripper_wait_s=1.0,
    ):
        super().__init__(robot, arm, gripper_wait_s)
        self.approach_offset = approach_offset
        self.grasp_offset = grasp_offset
        self.lift_height = lift_height

    def start(self, object_position, orientation=None):
        object_position = as_position(object_position)

        approach = object_position.copy()
        approach[2] += self.approach_offset

        grasp = object_position.copy()
        grasp[2] += self.grasp_offset

        lift = grasp.copy()
        lift[2] += self.lift_height

        self.targets = {
            "approach": approach,
            "grasp": grasp,
            "lift": lift,
            "orientation": None if orientation is None else as_quaternion(orientation),
        }
        self._enter_state("APPROACH")

    def update(self):
        if self.state == "IDLE":
            raise RuntimeError("PickStateMachine.update() called before start()")
        if self.done:
            return True

        orientation = self.targets["orientation"]

        if self.state == "APPROACH":
            self.robot.set_gripper(self.arm, True)
            self.robot.move_to(self.arm, self.targets["approach"], orientation)
            if self.robot.command_reached(self.arm, self.targets["approach"], orientation):
                self._enter_state("DOWN")

        elif self.state == "DOWN":
            self.robot.move_to(self.arm, self.targets["grasp"], orientation)
            if self.robot.reached(self.arm, self.targets["grasp"], orientation):
                self._enter_state("CLOSE")

        elif self.state == "CLOSE":
            self.robot.set_gripper(self.arm, False)
            if self.state_count >= self.gripper_wait_steps:
                self._enter_state("LIFT")

        elif self.state == "LIFT":
            self.robot.set_gripper(self.arm, False)
            self.robot.move_to(self.arm, self.targets["lift"], orientation)
            if self.robot.reached(self.arm, self.targets["lift"], orientation):
                self._enter_state("DONE")

        self._tick()
        return self.done


class PlaceStateMachine(_ArmStateMachine):
    """Move ABOVE the target, DOWN to release height, OPEN, RETREAT."""

    def __init__(
        self,
        robot,
        arm=LEFT,
        above_offset=0.15,
        release_offset=-0.07,
        gripper_wait_s=1.0,
    ):
        super().__init__(robot, arm, gripper_wait_s)
        self.above_offset = above_offset
        self.release_offset = release_offset

    def start(self, target_position, orientation=None):
        target_position = as_position(target_position)

        above = target_position.copy()
        above[2] += self.above_offset

        release = target_position.copy()
        release[2] += self.release_offset

        self.targets = {
            "above": above,
            "release": release,
            "orientation": None if orientation is None else as_quaternion(orientation),
        }
        self._enter_state("ABOVE")

    def update(self):
        if self.state == "IDLE":
            raise RuntimeError("PlaceStateMachine.update() called before start()")
        if self.done:
            return True

        orientation = self.targets["orientation"]

        if self.state == "ABOVE":
            self.robot.set_gripper(self.arm, False)
            self.robot.move_to(self.arm, self.targets["above"], orientation)
            if self.robot.command_reached(self.arm, self.targets["above"], orientation):
                self._enter_state("DOWN")

        elif self.state == "DOWN":
            self.robot.set_gripper(self.arm, False)
            self.robot.move_to(self.arm, self.targets["release"], orientation)
            if self.robot.reached(self.arm, self.targets["release"], orientation):
                self._enter_state("OPEN")

        elif self.state == "OPEN":
            self.robot.set_gripper(self.arm, True)
            if self.state_count >= self.gripper_wait_steps:
                self._enter_state("RETREAT")

        elif self.state == "RETREAT":
            self.robot.set_gripper(self.arm, True)
            self.robot.move_to(self.arm, self.targets["above"], orientation)
            if self.robot.reached(self.arm, self.targets["above"], orientation):
                self._enter_state("DONE")

        self._tick()
        return self.done


class HandoverStateMachine(_ArmStateMachine):
    """Pass an object held by ``giver_arm`` to ``receiver_arm``.

    BRING     giver carries the object to the handover pose while the
              receiver (gripper open) waits at handover + approach_offset.
    ALIGN     receiver moves in to handover + grasp_offset.
    GRASP     receiver closes; both grippers hold the object.
    RELEASE   giver opens.
    RETREAT   giver backs away to handover + retreat_offset. Done.

    ``approach_offset`` / ``retreat_offset`` default to lateral offsets that
    assume the receiver is the right arm (approaches from -y) and the giver
    the left arm (retreats toward +y). Override them for other geometries.
    These defaults are a sim scaffold; ``main.py`` disables real handover until
    calibrated, non-overlapping giver/receiver poses are supplied.
    """

    def __init__(
        self,
        robot,
        giver_arm=LEFT,
        receiver_arm=RIGHT,
        approach_offset=(0.0, -0.15, 0.0),
        grasp_offset=(0.0, 0.0, 0.0),
        retreat_offset=(0.0, 0.15, 0.0),
        gripper_wait_s=1.0,
    ):
        super().__init__(robot, giver_arm, gripper_wait_s)
        self.giver_arm = giver_arm
        self.receiver_arm = receiver_arm
        self.approach_offset = as_position(approach_offset)
        self.grasp_offset = as_position(grasp_offset)
        self.retreat_offset = as_position(retreat_offset)

    def start(self, handover_position, giver_orientation, receiver_orientation):
        handover_position = as_position(handover_position)
        self.targets = {
            "handover": handover_position,
            "standby": handover_position + self.approach_offset,
            "grasp": handover_position + self.grasp_offset,
            "retreat": handover_position + self.retreat_offset,
            "giver_orientation": as_quaternion(giver_orientation),
            "receiver_orientation": as_quaternion(receiver_orientation),
        }
        self._enter_state("BRING")

    def update(self):
        if self.state == "IDLE":
            raise RuntimeError("HandoverStateMachine.update() called before start()")
        if self.done:
            return True

        giver_orientation = self.targets["giver_orientation"]
        receiver_orientation = self.targets["receiver_orientation"]

        if self.state == "BRING":
            self.robot.set_gripper(self.giver_arm, False)
            self.robot.set_gripper(self.receiver_arm, True)
            self.robot.move_to(self.giver_arm, self.targets["handover"], giver_orientation)
            self.robot.move_to(self.receiver_arm, self.targets["standby"], receiver_orientation)
            if self.robot.reached(
                self.giver_arm, self.targets["handover"], giver_orientation
            ) and self.robot.command_reached(
                self.receiver_arm, self.targets["standby"], receiver_orientation
            ):
                self._enter_state("ALIGN")

        elif self.state == "ALIGN":
            self.robot.move_to(self.receiver_arm, self.targets["grasp"], receiver_orientation)
            if self.robot.reached(self.receiver_arm, self.targets["grasp"], receiver_orientation):
                self._enter_state("GRASP")

        elif self.state == "GRASP":
            self.robot.set_gripper(self.receiver_arm, False)
            if self.state_count >= self.gripper_wait_steps:
                self._enter_state("RELEASE")

        elif self.state == "RELEASE":
            self.robot.set_gripper(self.giver_arm, True)
            if self.state_count >= self.gripper_wait_steps:
                self._enter_state("RETREAT")

        elif self.state == "RETREAT":
            self.robot.move_to(self.giver_arm, self.targets["retreat"], giver_orientation)
            if self.robot.reached(self.giver_arm, self.targets["retreat"], giver_orientation):
                self._enter_state("DONE")

        self._tick()
        return self.done
