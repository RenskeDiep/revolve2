import math
import random
import numpy as np
from pyrr import Vector3
from revolve2.simulators.mujoco_simulator import LocalSimulator
from project2.torus_simulation_handler import TorusSimulationTeleportationHandler
import logging
from project2.utils.field_limits import FieldLimits
from revolve2.standards.morphological_measures import MorphologicalMeasures
import json


def initialize_local_simulator(
    plane_size: float, headless: bool = False, num_simulators: int = 1
) -> LocalSimulator:
    simulator = LocalSimulator(
        viewer_type="native", headless=headless, num_simulators=num_simulators
    )
    torus_handler = TorusSimulationTeleportationHandler(plane_size=plane_size)
    simulator.register_teleport_handler(torus_handler)
    logging.info(
        f"Registered teleportation handler with plane size: {plane_size}, half size: {torus_handler.half_size}"
    )
    return simulator


def get_random_free_position(
    limits: FieldLimits,
    existing_positions: list[Vector3],
    min_dist: float = 2.0,
    bailout_limit=1000,
) -> Vector3:
    """Find a random position that is not too close to existing robots."""
    for _ in range(bailout_limit):
        x = random.uniform(limits.get_x_min(), limits.get_x_max())
        y = random.uniform(limits.get_y_min(), limits.get_y_max())
        z = 0.0  # Assuming a flat field

        # Check distance from existing robots
        if all(
            math.sqrt((x - ex) ** 2 + (y - ey) ** 2) >= min_dist
            for ex, ey, _ in existing_positions
        ):
            return Vector3([x, y, z])  # Valid position found

    # Return some random location if we can't find a valid one
    return Vector3(
        [
            random.uniform(limits.get_x_min(), limits.get_x_max()),
            random.uniform(limits.get_y_min(), limits.get_y_max()),
            0,
        ]
    )

def similarity_score(individual1, individual2, normalize=True,
    method='euclidean'
)-> float:
    measures1 = MorphologicalMeasures(individual1.robot.body)
    v1 = np.array([
        measures1.num_modules,
        measures1.num_bricks,
        measures1.branching,
        measures1.limbs,
        measures1.length_of_limbs,
        measures1.coverage,
        measures1.proportion_2d if measures1.is_2d else 0.0,
        measures1.symmetry,
    ], dtype=float)
    measures2 = MorphologicalMeasures(individual2.robot.body)
    v2 = np.array([
        measures2.num_modules,
        measures2.num_bricks,
        measures2.branching,
        measures2.limbs,
        measures2.length_of_limbs,
        measures2.coverage,
        measures2.proportion_2d if measures2.is_2d else 0.0,
        measures2.symmetry,
    ], dtype=float)

 
    if normalize:
        max_vals = np.maximum(v1, v2)
        max_vals[max_vals == 0] = 1.0  # avoid divide-by-zero
        v1 /= max_vals
        v2 /= max_vals

    if method == 'euclidean':
        euclidean = np.linalg.norm(v1 - v2)
        max_dist = np.sqrt(len(v1))  # Max possible distance in normalized space
        return 1 - (euclidean / max_dist)
    elif method == 'cosine':
        dot = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        return 1 - (dot / norm_product if norm_product != 0 else 0.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    

def save_dict_to_json(data_dict, filename="data.json"):
    with open(filename, "w+") as f:
        json.dump(data_dict, f, indent=4)


