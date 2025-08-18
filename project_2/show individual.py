"""Main script for the example."""

import logging
from project2.config import Config
from uuid import UUID

from pyrr import Vector3
from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.simulation.scene.vector2.vector2 import Vector2

import multineat

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulation.scene import Pose
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from project2.utils.helpers import similarity_score
from itertools import combinations

# from revolve2.standards.mate_selection import Reproducer
import math
# import random


from project2.individual import Individual, reproduce as reproduce_individual
from project2.incubator import Incubator
from project2.utils.helpers import initialize_local_simulator, get_random_free_position
from project2.simulation_result import SimulationResult, FitnessFunctionAlgorithm
from project2.stats import Statistics
import project2.mate_selection as mate_selection
from project2.death_mechanism import apply_death_mechanism
from revolve2.standards.morphological_measures import MorphologicalMeasures
import numpy as np
from collections import defaultdict
from project2.utils.helpers import save_dict_to_json
from project2.genotype import Genotype
import pickle
import codecs
import numpy as np

from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.standards import fitness_functions, modular_robots_v2, terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters



def main(config: Config, folder_name: str = "stats") -> None:
    """Run the simulation."""
    
    # Set up the random number generator.
    rng = make_rng_time_seed()
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    plane_size = config.LIMITS.calculate_plane_size()

    genotype = {'body': 'gASVsgsAAAAAAACMR3Jldm9sdmUyLnN0YW5kYXJkcy5nZW5vdHlwZXMuY3Bwbndpbi5fbXVsdGlu\nZWF0X2dlbm90eXBlX3BpY2tsZV93cmFwcGVylIweTXVsdGluZWF0R2Vub3R5cGVQaWNrbGVXcmFw\ncGVylJOUKYGUWDoLAAB7CiJ2YWx1ZTAiOnsKInZhbHVlMCI6MCwKInZhbHVlMSI6Wwp7CiJ2YWx1\nZTAiOnsKInZhbHVlMCI6W10KfSwKInZhbHVlMSI6MSwKInZhbHVlMiI6MSwKInZhbHVlMyI6MC4w\nLAoidmFsdWU0IjowLjAsCiJ2YWx1ZTUiOjAuMCwKInZhbHVlNiI6MC4wLAoidmFsdWU3IjowLAoi\ndmFsdWU4IjowLAoidmFsdWU5IjoxLAoidmFsdWUxMCI6MC4wCn0sCnsKInZhbHVlMCI6ewoidmFs\ndWUwIjpbXQp9LAoidmFsdWUxIjoyLAoidmFsdWUyIjoxLAoidmFsdWUzIjowLjAsCiJ2YWx1ZTQi\nOjAuMCwKInZhbHVlNSI6MC4wLAoidmFsdWU2IjowLjAsCiJ2YWx1ZTciOjAsCiJ2YWx1ZTgiOjAs\nCiJ2YWx1ZTkiOjEsCiJ2YWx1ZTEwIjowLjAKfSwKewoidmFsdWUwIjp7CiJ2YWx1ZTAiOltdCn0s\nCiJ2YWx1ZTEiOjMsCiJ2YWx1ZTIiOjEsCiJ2YWx1ZTMiOjAuMCwKInZhbHVlNCI6MC4wLAoidmFs\ndWU1IjowLjAsCiJ2YWx1ZTYiOjAuMCwKInZhbHVlNyI6MCwKInZhbHVlOCI6MCwKInZhbHVlOSI6\nMSwKInZhbHVlMTAiOjAuMAp9LAp7CiJ2YWx1ZTAiOnsKInZhbHVlMCI6W10KfSwKInZhbHVlMSI6\nNCwKInZhbHVlMiI6MSwKInZhbHVlMyI6MC4wLAoidmFsdWU0IjowLjAsCiJ2YWx1ZTUiOjAuMCwK\nInZhbHVlNiI6MC4wLAoidmFsdWU3IjowLAoidmFsdWU4IjowLAoidmFsdWU5IjoxLAoidmFsdWUx\nMCI6MC4wCn0sCnsKInZhbHVlMCI6ewoidmFsdWUwIjpbXQp9LAoidmFsdWUxIjo1LAoidmFsdWUy\nIjoyLAoidmFsdWUzIjowLjAsCiJ2YWx1ZTQiOjAuMCwKInZhbHVlNSI6MC4wLAoidmFsdWU2Ijow\nLjAsCiJ2YWx1ZTciOjAsCiJ2YWx1ZTgiOjAsCiJ2YWx1ZTkiOjEsCiJ2YWx1ZTEwIjowLjAKfSwK\newoidmFsdWUwIjp7CiJ2YWx1ZTAiOltdCn0sCiJ2YWx1ZTEiOjYsCiJ2YWx1ZTIiOjQsCiJ2YWx1\nZTMiOjMuMDI1LAoidmFsdWU0IjowLjAsCiJ2YWx1ZTUiOjAuMCwKInZhbHVlNiI6MC4wLAoidmFs\ndWU3IjowLAoidmFsdWU4IjowLAoidmFsdWU5IjoxMCwKInZhbHVlMTAiOjEuMAp9LAp7CiJ2YWx1\nZTAiOnsKInZhbHVlMCI6W10KfSwKInZhbHVlMSI6NywKInZhbHVlMiI6NCwKInZhbHVlMyI6My4w\nMjUsCiJ2YWx1ZTQiOjAuMCwKInZhbHVlNSI6MC4wLAoidmFsdWU2IjowLjAsCiJ2YWx1ZTciOjAs\nCiJ2YWx1ZTgiOjAsCiJ2YWx1ZTkiOjEwLAoidmFsdWUxMCI6MS4wCn0sCnsKInZhbHVlMCI6ewoi\ndmFsdWUwIjpbXQp9LAoidmFsdWUxIjoyLAoidmFsdWUyIjozLAoidmFsdWUzIjoxLjg1MDU1MDkw\nNzIxMDE3MjMsCiJ2YWx1ZTQiOjAuMCwKInZhbHVlNSI6MC4wLAoidmFsdWU2IjowLjAsCiJ2YWx1\nZTciOjAsCiJ2YWx1ZTgiOjAsCiJ2YWx1ZTkiOjYsCiJ2YWx1ZTEwIjowLjUKfQpdLAoidmFsdWUy\nIjpbCnsKInZhbHVlMCI6ewoidmFsdWUwIjpbXQp9LAoidmFsdWUxIjoxLAoidmFsdWUyIjo2LAoi\ndmFsdWUzIjoxLAoidmFsdWU0IjpmYWxzZSwKInZhbHVlNSI6MC4wMDEzMDg3Nzc5OTMyMjQ4MzI5\nCn0sCnsKInZhbHVlMCI6ewoidmFsdWUwIjpbXQp9LAoidmFsdWUxIjoyLAoidmFsdWUyIjo2LAoi\ndmFsdWUzIjoyLAoidmFsdWU0IjpmYWxzZSwKInZhbHVlNSI6MC4zMTM4Njg0MzMzNzgzNDg4Cn0s\nCnsKInZhbHVlMCI6ewoidmFsdWUwIjpbXQp9LAoidmFsdWUxIjozLAoidmFsdWUyIjo2LAoidmFs\ndWUzIjozLAoidmFsdWU0IjpmYWxzZSwKInZhbHVlNSI6MC42NzI3NjM2MjM2NDAyNjE5Cn0sCnsK\nInZhbHVlMCI6ewoidmFsdWUwIjpbXQp9LAoidmFsdWUxIjo0LAoidmFsdWUyIjo2LAoidmFsdWUz\nIjo0LAoidmFsdWU0IjpmYWxzZSwKInZhbHVlNSI6MC4zOTMyNDQyMjE2MzI1MTgwNwp9LAp7CiJ2\nYWx1ZTAiOnsKInZhbHVlMCI6W10KfSwKInZhbHVlMSI6NSwKInZhbHVlMiI6NiwKInZhbHVlMyI6\nNSwKInZhbHVlNCI6ZmFsc2UsCiJ2YWx1ZTUiOjAuMjcwNTg5NTkxODEwNTkwNwp9LAp7CiJ2YWx1\nZTAiOnsKInZhbHVlMCI6W10KfSwKInZhbHVlMSI6MSwKInZhbHVlMiI6NywKInZhbHVlMyI6NiwK\nInZhbHVlNCI6ZmFsc2UsCiJ2YWx1ZTUiOi0wLjYzODQ1NzA0NDY4MTc4MDMKfSwKewoidmFsdWUw\nIjp7CiJ2YWx1ZTAiOltdCn0sCiJ2YWx1ZTEiOjIsCiJ2YWx1ZTIiOjcsCiJ2YWx1ZTMiOjcsCiJ2\nYWx1ZTQiOmZhbHNlLAoidmFsdWU1IjowLjMyMDk5MDA3MTQ3NTEzOTc3Cn0sCnsKInZhbHVlMCI6\newoidmFsdWUwIjpbXQp9LAoidmFsdWUxIjozLAoidmFsdWUyIjo3LAoidmFsdWUzIjo4LAoidmFs\ndWU0IjpmYWxzZSwKInZhbHVlNSI6LTAuMDA1MTczNTkwOTIwNTcwODc1Cn0sCnsKInZhbHVlMCI6\newoidmFsdWUwIjpbXQp9LAoidmFsdWUxIjo1LAoidmFsdWUyIjo3LAoidmFsdWUzIjoxMCwKInZh\nbHVlNCI6ZmFsc2UsCiJ2YWx1ZTUiOjAuMjA3NTkyMzE5MDc2MTU5MTQKfSwKewoidmFsdWUwIjp7\nCiJ2YWx1ZTAiOltdCn0sCiJ2YWx1ZTEiOjQsCiJ2YWx1ZTIiOjIsCiJ2YWx1ZTMiOjYsCiJ2YWx1\nZTQiOmZhbHNlLAoidmFsdWU1IjotMS4xMzA0NzUyOTE0MTUwMzYzCn0sCnsKInZhbHVlMCI6ewoi\ndmFsdWUwIjpbXQp9LAoidmFsdWUxIjoyLAoidmFsdWUyIjo3LAoidmFsdWUzIjo3LAoidmFsdWU0\nIjpmYWxzZSwKInZhbHVlNSI6MC4xNjMxNTM0Mjc2MDA4MTc1Mwp9LAp7CiJ2YWx1ZTAiOnsKInZh\nbHVlMCI6W10KfSwKInZhbHVlMSI6NCwKInZhbHVlMiI6NywKInZhbHVlMyI6OCwKInZhbHVlNCI6\nZmFsc2UsCiJ2YWx1ZTUiOi0wLjI3NjMwMjk3MDQxMDEwNjU2Cn0KXSwKInZhbHVlMyI6NSwKInZh\nbHVlNCI6MiwKInZhbHVlNSI6MC4wLAoidmFsdWU2IjowLjAsCiJ2YWx1ZTciOjAsCiJ2YWx1ZTgi\nOjAuMCwKInZhbHVlOSI6ZmFsc2UsCiJ2YWx1ZTEwIjoxNjM4NCwKInZhbHVlMTEiOnsKInZhbHVl\nMCI6W10KfSwKInZhbHVlMTIiOjcsCiJ2YWx1ZTEzIjoxMAp9Cn2UYi4=\n', 'brain': 'gASVbgkAAAAAAACMR3Jldm9sdmUyLnN0YW5kYXJkcy5nZW5vdHlwZXMuY3Bwbndpbi5fbXVsdGlu\nZWF0X2dlbm90eXBlX3BpY2tsZV93cmFwcGVylIweTXVsdGluZWF0R2Vub3R5cGVQaWNrbGVXcmFw\ncGVylJOUKYGUWPYIAAB7CiJ2YWx1ZTAiOnsKInZhbHVlMCI6MCwKInZhbHVlMSI6Wwp7CiJ2YWx1\nZTAiOnsKInZhbHVlMCI6W10KfSwKInZhbHVlMSI6MSwKInZhbHVlMiI6MSwKInZhbHVlMyI6MC4w\nLAoidmFsdWU0IjowLjAsCiJ2YWx1ZTUiOjAuMCwKInZhbHVlNiI6MC4wLAoidmFsdWU3IjowLAoi\ndmFsdWU4IjowLAoidmFsdWU5IjoxLAoidmFsdWUxMCI6MC4wCn0sCnsKInZhbHVlMCI6ewoidmFs\ndWUwIjpbXQp9LAoidmFsdWUxIjoyLAoidmFsdWUyIjoxLAoidmFsdWUzIjowLjAsCiJ2YWx1ZTQi\nOjAuMCwKInZhbHVlNSI6MC4wLAoidmFsdWU2IjowLjAsCiJ2YWx1ZTciOjAsCiJ2YWx1ZTgiOjAs\nCiJ2YWx1ZTkiOjEsCiJ2YWx1ZTEwIjowLjAKfSwKewoidmFsdWUwIjp7CiJ2YWx1ZTAiOltdCn0s\nCiJ2YWx1ZTEiOjMsCiJ2YWx1ZTIiOjEsCiJ2YWx1ZTMiOjAuMCwKInZhbHVlNCI6MC4wLAoidmFs\ndWU1IjowLjAsCiJ2YWx1ZTYiOjAuMCwKInZhbHVlNyI6MCwKInZhbHVlOCI6MCwKInZhbHVlOSI6\nMSwKInZhbHVlMTAiOjAuMAp9LAp7CiJ2YWx1ZTAiOnsKInZhbHVlMCI6W10KfSwKInZhbHVlMSI6\nNCwKInZhbHVlMiI6MSwKInZhbHVlMyI6MC4wLAoidmFsdWU0IjowLjAsCiJ2YWx1ZTUiOjAuMCwK\nInZhbHVlNiI6MC4wLAoidmFsdWU3IjowLAoidmFsdWU4IjowLAoidmFsdWU5IjoxLAoidmFsdWUx\nMCI6MC4wCn0sCnsKInZhbHVlMCI6ewoidmFsdWUwIjpbXQp9LAoidmFsdWUxIjo1LAoidmFsdWUy\nIjoxLAoidmFsdWUzIjowLjAsCiJ2YWx1ZTQiOjAuMCwKInZhbHVlNSI6MC4wLAoidmFsdWU2Ijow\nLjAsCiJ2YWx1ZTciOjAsCiJ2YWx1ZTgiOjAsCiJ2YWx1ZTkiOjEsCiJ2YWx1ZTEwIjowLjAKfSwK\newoidmFsdWUwIjp7CiJ2YWx1ZTAiOltdCn0sCiJ2YWx1ZTEiOjYsCiJ2YWx1ZTIiOjEsCiJ2YWx1\nZTMiOjAuMCwKInZhbHVlNCI6MC4wLAoidmFsdWU1IjowLjAsCiJ2YWx1ZTYiOjAuMCwKInZhbHVl\nNyI6MCwKInZhbHVlOCI6MCwKInZhbHVlOSI6MSwKInZhbHVlMTAiOjAuMAp9LAp7CiJ2YWx1ZTAi\nOnsKInZhbHVlMCI6W10KfSwKInZhbHVlMSI6NywKInZhbHVlMiI6MiwKInZhbHVlMyI6MC4wLAoi\ndmFsdWU0IjowLjAsCiJ2YWx1ZTUiOjAuMCwKInZhbHVlNiI6MC4wLAoidmFsdWU3IjowLAoidmFs\ndWU4IjowLAoidmFsdWU5IjoxLAoidmFsdWUxMCI6MC4wCn0sCnsKInZhbHVlMCI6ewoidmFsdWUw\nIjpbXQp9LAoidmFsdWUxIjo4LAoidmFsdWUyIjo0LAoidmFsdWUzIjozLjAyNSwKInZhbHVlNCI6\nMC4wLAoidmFsdWU1IjowLjAsCiJ2YWx1ZTYiOjAuMCwKInZhbHVlNyI6MCwKInZhbHVlOCI6MCwK\nInZhbHVlOSI6OSwKInZhbHVlMTAiOjEuMAp9Cl0sCiJ2YWx1ZTIiOlsKewoidmFsdWUwIjp7CiJ2\nYWx1ZTAiOltdCn0sCiJ2YWx1ZTEiOjEsCiJ2YWx1ZTIiOjgsCiJ2YWx1ZTMiOjEsCiJ2YWx1ZTQi\nOmZhbHNlLAoidmFsdWU1IjotMC43MTQxMjE4MzI1MTk5NDk5Cn0sCnsKInZhbHVlMCI6ewoidmFs\ndWUwIjpbXQp9LAoidmFsdWUxIjoyLAoidmFsdWUyIjo4LAoidmFsdWUzIjoyLAoidmFsdWU0Ijpm\nYWxzZSwKInZhbHVlNSI6MC4wNTA1ODI2NDY4NzgxODIxNzgKfSwKewoidmFsdWUwIjp7CiJ2YWx1\nZTAiOltdCn0sCiJ2YWx1ZTEiOjMsCiJ2YWx1ZTIiOjgsCiJ2YWx1ZTMiOjMsCiJ2YWx1ZTQiOmZh\nbHNlLAoidmFsdWU1IjowLjM5MjQwNDU5NjgwMTAxMTY2Cn0sCnsKInZhbHVlMCI6ewoidmFsdWUw\nIjpbXQp9LAoidmFsdWUxIjo0LAoidmFsdWUyIjo4LAoidmFsdWUzIjo0LAoidmFsdWU0IjpmYWxz\nZSwKInZhbHVlNSI6LTAuMDExMjI1OTY3NTEwNjU1MDc4Cn0sCnsKInZhbHVlMCI6ewoidmFsdWUw\nIjpbXQp9LAoidmFsdWUxIjo1LAoidmFsdWUyIjo4LAoidmFsdWUzIjo1LAoidmFsdWU0IjpmYWxz\nZSwKInZhbHVlNSI6MC45NTkzMTQ5NTE4MDYxNDk3Cn0sCnsKInZhbHVlMCI6ewoidmFsdWUwIjpb\nXQp9LAoidmFsdWUxIjo2LAoidmFsdWUyIjo4LAoidmFsdWUzIjo2LAoidmFsdWU0IjpmYWxzZSwK\nInZhbHVlNSI6LTAuMDkwMTQ1OTAxNTI2MjgzMzIKfSwKewoidmFsdWUwIjp7CiJ2YWx1ZTAiOltd\nCn0sCiJ2YWx1ZTEiOjcsCiJ2YWx1ZTIiOjgsCiJ2YWx1ZTMiOjcsCiJ2YWx1ZTQiOmZhbHNlLAoi\ndmFsdWU1IjotMC41OTIwNzIwNzMzMDU4ODczCn0KXSwKInZhbHVlMyI6NywKInZhbHVlNCI6MSwK\nInZhbHVlNSI6MC4wLAoidmFsdWU2IjowLjAsCiJ2YWx1ZTciOjAsCiJ2YWx1ZTgiOjAuMCwKInZh\nbHVlOSI6ZmFsc2UsCiJ2YWx1ZTEwIjoxNjM4NCwKInZhbHVlMTEiOnsKInZhbHVlMCI6W10KfSwK\nInZhbHVlMTIiOjgsCiJ2YWx1ZTEzIjo3Cn0KfZRiLg==\n'}

    body = pickle.loads(codecs.decode(genotype["body"].encode(), "base64"))
    brain = pickle.loads(codecs.decode(genotype["brain"].encode(), "base64"))

# rebuild genotype
    genotype = Genotype(body=body, brain=brain)
    offspring = [genotype]
    #offspring = [Genotype(body=genotype["body"], brain=genotype["brain"])]
    ind = Individual(genotype=offspring[0], fitness=0.0)


    #robot = ind.develop(config.VISUALIZE_MAP)
    body = modular_robots_v2.spider_v2()
    brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
    robot = ModularRobot(body, brain)
    measures1 = MorphologicalMeasures(robot.body)
    v1 = np.array([
                measures1.num_modules,
                measures1.num_bricks,
                measures1.branching,
                measures1.limbs,
                measures1.length_of_limbs,
                measures1.coverage,
                measures1.symmetry,
            ], dtype=float)
    print(v1)


    # Now we can create a scene and add the robots by mapping the genotypes to phenotypes
    scene = ModularRobotScene(terrain=terrains.flat(Vector2([plane_size, plane_size])))
    initial_positions: list[Vector3] = []
    pos = [0,0,0]
    pose = Pose(pos)
    #pose.orientation = [360.0, 360.0, 0.0, 1.0]
    #get_random_free_position(config.LIMITS, initial_positions)
    scene.add_robot(robot, pose=pose)
    initial_positions.append(pos)
        
    simulation_results = []
    # Create the simulator.
    simulator = initialize_local_simulator(
        plane_size, headless=False, num_simulators=1
    )

    for generation in range(1):

        logging.info(f"Starting generation {generation}.")
        simulation_result_list = simulate_scenes(
            simulator=simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scene,
        )  # Process one scene at a time

        # TODO process each simulation state to simulate continuos mating
        simulation_result = SimulationResult(
            simulation_result_list,
            plane_size=plane_size,
            movement_weight=config.MOVEMENT_WEIGHT,
        )
        simulation_results.append(simulation_result)

        scene = ModularRobotScene(
            terrain=terrains.flat(Vector2([plane_size, plane_size]))
        )

        

if __name__ == "__main__":
    import argparse
    import os
    import glob

    # Dynamically get available config choices from configs/ folder
    config_dir = os.path.join(os.path.dirname(__file__), "project2", "configs")
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    available_configs = [os.path.splitext(os.path.basename(f))[0] for f in config_files]
    available_configs.sort()  # Sort for consistent ordering

    parser = argparse.ArgumentParser(
        description="Run simulation with different configs"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=available_configs,
        default="old/config1"
        if "config1" in available_configs
        else (available_configs[0] if available_configs else "config1"),
        help=f"Config name to use. Available configs: {', '.join(available_configs)}",
    )
    args = parser.parse_args()

    # Create a Config instance with the specified config name
    config = Config(args.config)

    main(config, f"stats/{args.config}")
