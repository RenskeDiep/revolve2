from pyrr import Vector3

from .._right_angles import RightAngles
from ..base import ActiveHinge
from ..sensors import ActiveHingeSensor


class ActiveHingeV2(ActiveHinge):
    """
    An active hinge v2 module for a modular robot.

    This is a rotary joint.
    """

    def __init__(self, rotation: float | RightAngles):
        """
        Initialize this object.

        :param rotation: The Modules rotation.
        """
        super().__init__(
            rotation=rotation,
            range=1.047197551,
            effort=0.948013269,
            velocity=6.338968228,
            frame_bounding_box=Vector3([0.018, 0.052, 0.0165891]),
            frame_offset=0.04495,
            servo1_bounding_box=Vector3([0.05125, 0.0512, 0.020]),
            servo2_bounding_box=Vector3([0.002, 0.052, 0.052]),
            frame_mass=0.01144,
            servo1_mass=0.058,
            servo2_mass=0.025,
            servo_offset=0.0239,
            joint_offset=0.0119,
            static_friction=1.0,
            dynamic_friction=1.0,
            armature=0.002,
            pid_gain_p=5.0,
            pid_gain_d=0.05,
            child_offset=0.05125 / 2 + 0.002 + 0.01,
            sensors=[
                ActiveHingeSensor()
            ],  # By default, V2 robots have ActiveHinge sensors, since the hardware also supports them natively.
        )
