from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg


_DONOR_ASSET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "isaac_drone_racer", "assets")
)
HUMMINGBIRD_USD_PATH = os.path.join(_DONOR_ASSET_DIR, "hummingbird.usd")


HUMMINGBIRD_PARAMS = {
    "name": "hummingbird",
    "mass": 0.716,
    "inertia": {
        "xx": 0.007,
        "yy": 0.007,
        "zz": 0.012,
    },
    "arm_length": 0.17,
    "drag_coef": 0.2,
    "rotor_configuration": {
        "num_rotors": 4,
        "arm_lengths": [0.17, 0.17, 0.17, 0.17],
        "directions": [-1.0, 1.0, -1.0, 1.0],
        "force_constants": [8.54858e-06] * 4,
        "moment_constants": [1.3677728816219314e-07] * 4,
        "max_rotation_velocities": [838, 838, 838, 838],
        "rotor_angles": [0, 1.57079632679, 3.14159265359, -1.57079632679],
    },
}


def get_hummingbird_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HUMMINGBIRD_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )


HUMMINGBIRD = get_hummingbird_cfg()
