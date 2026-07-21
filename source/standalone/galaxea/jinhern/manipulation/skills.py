"""Blocking skill functions built on the tick-based state machines.

Each skill constructs a state machine, then loops ``update()`` + ``step()``
until the skill reports done or the backend stops running. They return True
on success, False if the backend shut down mid-skill or ``timeout_s`` hit.
"""

import math

from .interface import LEFT, RIGHT
from .state_machine import HandoverStateMachine, PickStateMachine, PlaceStateMachine


def run_state_machine(robot, state_machine, timeout_s=None, log_every_s=0.0):
    """Drive a started state machine to completion. Returns True on success."""
    step = 0
    log_every_steps = (
        max(1, math.ceil(log_every_s / robot.step_dt)) if log_every_s > 0 else 0
    )
    max_steps = math.ceil(timeout_s / robot.step_dt) if timeout_s else None

    while robot.is_running:
        if state_machine.update():
            return True
        robot.step()
        step += 1

        if log_every_steps and step % log_every_steps == 0:
            print(f"[{type(state_machine).__name__}] state={state_machine.state}")
        if max_steps is not None and step >= max_steps:
            print(
                f"[{type(state_machine).__name__}] timed out after {timeout_s}s "
                f"in state {state_machine.state}"
            )
            robot.abort()
            return False
    robot.abort()
    return False


def pick(
    robot,
    object_position,
    orientation=None,
    arm=LEFT,
    timeout_s=None,
    log_every_s=2.0,
    **config,
):
    """Pick an object: approach from above, descend, close gripper, lift.

    Extra keyword arguments (``approach_offset``, ``grasp_offset``,
    ``lift_height``, ``gripper_wait_s``) are forwarded to PickStateMachine.
    """
    state_machine = PickStateMachine(robot, arm=arm, **config)
    state_machine.start(object_position, orientation)
    return run_state_machine(robot, state_machine, timeout_s, log_every_s)


def place(
    robot,
    target_position,
    orientation=None,
    arm=LEFT,
    timeout_s=None,
    log_every_s=2.0,
    **config,
):
    """Place the held object at a target position, then retreat upward."""
    state_machine = PlaceStateMachine(robot, arm=arm, **config)
    state_machine.start(target_position, orientation)
    return run_state_machine(robot, state_machine, timeout_s, log_every_s)


def handover(
    robot,
    handover_position,
    giver_orientation,
    receiver_orientation,
    giver_arm=LEFT,
    receiver_arm=RIGHT,
    timeout_s=None,
    log_every_s=2.0,
    **config,
):
    """Pass the object held by ``giver_arm`` to ``receiver_arm``."""
    state_machine = HandoverStateMachine(
        robot, giver_arm=giver_arm, receiver_arm=receiver_arm, **config
    )
    state_machine.start(handover_position, giver_orientation, receiver_orientation)
    return run_state_machine(robot, state_machine, timeout_s, log_every_s)
