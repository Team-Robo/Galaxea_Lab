"""Backend-agnostic R1 manipulation pipeline.

Lightweight modules (numpy only) are exported here. The backends have heavy
dependencies and must be imported explicitly:

    from manipulation.isaac_robot import IsaacManipulator   # after AppLauncher
    from manipulation.r1_robot import RealRobotManipulator  # needs rclpy
"""

from .interface import LEFT, RIGHT, ManipulatorInterface
from .state_machine import HandoverStateMachine, PickStateMachine, PlaceStateMachine
from .skills import handover, pick, place, run_state_machine

__all__ = [
    "LEFT",
    "RIGHT",
    "ManipulatorInterface",
    "PickStateMachine",
    "PlaceStateMachine",
    "HandoverStateMachine",
    "pick",
    "place",
    "handover",
    "run_state_machine",
]
