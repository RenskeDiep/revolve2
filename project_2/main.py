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


def main(config: Config, folder_name: str = "stats") -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()
    meeting = 0
    mating = 0
    similarity_scores = defaultdict(list)
    individual_count = defaultdict(list)
    fitness = defaultdict(list)
    novelty_child_population = defaultdict(list)
    novelty_child_parents = defaultdict(list)
    coordinates_per_gen = defaultdict(list)
    measures = defaultdict(list)
    uuid_to_measures = defaultdict(list)
    gen_to_robots = defaultdict(list)

    stats = Statistics(folder_name=folder_name)

    # Set up the random number generator.
    rng = make_rng_time_seed()
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    plane_size = config.LIMITS.calculate_plane_size()

    # Create an initial population, with pre-trained brains
    logging.info("Generating initial population.")
    population = Incubator(
        population_size=config.POPULATION_SIZE,
        training_budget=config.INCUBATOR_TRAINING_BUDGET,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
        num_simulators=config.NUM_SIMULATORS,
    ).incubate()

    uuid_to_robot: dict[str, ModularRobot] = {}
    uuid_to_individual: dict[str, Individual] = {}
    for ind in population:
        robot = ind.develop(config.VISUALIZE_MAP)
        uuid_to_robot[ind.get_robot_uuid()] = robot
        uuid_to_individual[ind.get_robot_uuid()] = ind
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
        uuid_to_measures[str(robot.uuid)].append(v1.tolist())


    # Now we can create a scene and add the robots by mapping the genotypes to phenotypes
    scene = ModularRobotScene(terrain=terrains.flat(Vector2([plane_size, plane_size])))
    initial_positions: list[Vector3] = []

    logging.info("Adding initial robots to scene.")
    for robot in list(uuid_to_robot.values()):
        pos = get_random_free_position(config.LIMITS, initial_positions)
        scene.add_robot(robot, pose=Pose(pos))
        initial_positions.append(pos)
        
    simulation_results = []
    # Create the simulator.
    simulator = initialize_local_simulator(
        plane_size, headless=config.SIMULATION_HEADLESS, num_simulators=1
    )
    met_before: dict[tuple, int] = {}

    for generation in range(config.ITERATIONS):

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

        current_robots = [robot for robot, _, _ in scene._robots]
        fitness_values = simulation_result.fitness(
            current_robots, config.FITNESS_FUNCTION_ALGORITHM
        )

        # Get additional fitness metrics for stats tracking
        all_fitness_metrics = simulation_result.get_all_fitness_metrics(current_robots)

        # Map fitness metrics to robot UUIDs
        fitness_metrics_by_uuid = {}
        for metric_name, metric_values in all_fitness_metrics.items():
            fitness_metrics_by_uuid[metric_name] = {}
            for robot, metric_value in zip(current_robots, metric_values):
                if robot.uuid in uuid_to_individual:
                    fitness_metrics_by_uuid[metric_name][robot.uuid] = metric_value

        for robot, fitness in zip(current_robots, fitness_values):
            if robot.uuid in uuid_to_individual:
                uuid_to_individual[robot.uuid].fitness = fitness

        stats.track_individuals(uuid_to_individual, generation, fitness_metrics_by_uuid)
        stats.add_generation(generation, len(population))
        stats.flush_to_json(f"generation_{generation}")

        existing_robots: list[ModularRobot] = []
        existing_robots_uuids: set[UUID] = set()
        existing_positions: list[Vector3] = []
        coordinates: list[tuple[float, float, float]] = []
        final_coordinates: list[tuple[float, float, float]] = []
        for robot, pose, _ in scene._robots:
            existing_robots.append(robot)
            existing_robots_uuids.add(robot.uuid)
            existing_positions.append(pose.position)
            states = simulation_result.get_scene_states()
            state_id = 0
            for state in states:
                state_id += 1
                xyz = (
                    state.get_modular_robot_simulation_state(robot).get_pose().position
                )
                coordinates.append((xyz.x, xyz.y, xyz.z, robot, state_id))
            final_state = simulation_result.get_final_scene_state()
            final_xyz = (
                final_state.get_modular_robot_simulation_state(robot)
                .get_pose()
                .position
            )
            final_coordinates.append((xyz.x, xyz.y, xyz.z))
            coordinates_per_gen[generation].append((xyz.x, xyz.y, xyz.z, str(robot.uuid)))

        
        #if generation % 5 == 0:
        #    sim_score_list = []
        #    for robot1 in current_robots:
        #        for robot2 in current_robots:
        #            if robot1.uuid != robot2.uuid: 
        #                individual1 = uuid_to_individual[robot1.uuid]
        #                individual2 = uuid_to_individual[robot2.uuid]
        #                sim_score = similarity_score(individual1, individual2)
        #                sim_score_list.append(sim_score)

        #    similarity_scores[generation] = sim_score_list

        individual_count[generation] = len(current_robots)

        for robot in current_robots:
            gen_to_robots[generation].append(str(robot.uuid))

        logging.info(f"coordinates length: {len(coordinates)}")
        logging.info(f"existing_robots length: {len(existing_robots)}")
        logging.info(f"existing_positions length: {len(existing_positions)}")

        scene = ModularRobotScene(
            terrain=terrains.flat(Vector2([plane_size, plane_size]))
        )

        # TOOD: reafactor this so it's easier to read
        for (i, (x1, y1, z1, robot1, state_id1)), (
            j,
            (x2, y2, z2, robot2, state_id2),
        ) in combinations(enumerate(coordinates), 2):
            individual1 = uuid_to_individual[robot1.uuid]
            individual2 = uuid_to_individual[robot2.uuid]
            if robot1 != robot2 and state_id1 == state_id2:
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                if distance <= config.MATING_THRESHOLD:
                    r1_uuid = robot1.uuid
                    r2_uuid = robot2.uuid
                    # r1_uuid = existing_robots[i].uuid
                    # r2_uuid = existing_robots[j].uuid
                    pair = tuple(sorted((r1_uuid, r2_uuid)))
                    if (
                        pair not in met_before
                        or generation - met_before[pair] >= config.MATING_COOLDOWN
                    ):
                        met_before[pair] = generation
                        logging.info(
                            f"Meeting: Robots {i} and {j} - Distance: {distance:.3f}"
                        )
                        meeting += 1
                        if mate_selection.mate_decision(
                            config.MATE_SELECTION_STRATEGY,
                            config.SIMILARITY_THRES_MIN, 
                            config.SIMILARITY_THRES_MAX,
                            individual1,
                            individual2,
                            population,
                            config.MATE_SELECTION_THRESHOLD,  
                        ):
                            logging.info("YAY mating!")
                            mating += 1

                            # Increment offspring count for both parents
                            stats.increment_offspring_count(r1_uuid)
                            stats.increment_offspring_count(r2_uuid)

                            offspring = reproduce_individual(
                                individual1,
                                individual2,
                                rng,
                                innov_db_body,
                                innov_db_brain,
                                generation,
                            )
                            offspring_robot = offspring.develop(config.VISUALIZE_MAP)
                            population.append(offspring)
                            uuid_to_individual[offspring_robot.uuid] = offspring
                            uuid_to_robot[offspring_robot.uuid] = offspring_robot
                            measures1 = MorphologicalMeasures(offspring_robot.body)
                            v1 = np.array([
                            measures1.num_modules,
                            measures1.num_bricks,
                            measures1.branching,
                            measures1.limbs,
                            measures1.length_of_limbs,
                            measures1.coverage,
                            measures1.symmetry,
                            ], dtype=float)
                            uuid_to_measures[str(offspring_robot.uuid)].append(v1.tolist())


                            sim_offspring = uuid_to_individual[offspring_robot.uuid]
                            for robot2 in current_robots:
                                sim2 = uuid_to_individual[robot2.uuid]
                                nov_score = similarity_score(sim_offspring, sim2)
                                novelty_child_population[str(offspring_robot.uuid)].append((str(robot2.uuid), nov_score, generation))

                            parent_score1 = similarity_score(sim_offspring, individual1)
                            parent_score2 = similarity_score(sim_offspring, individual2)
                            novelty_child_parents[str(offspring_robot.uuid)].append((parent_score1, parent_score2, generation))

        with open("C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/run 1/extra.txt", "a") as f:
            f.write("Generation: " + str(generation) + "\n")
            f.write("Mating: " + str(mating) + "\n")
            f.write("Meeting: " + str(meeting) + "\n")
            f.write("Individual count: " + str(individual_count) + "\n")

        # Apply death mechanism based on configuration
        dead_individuals = apply_death_mechanism(
            population=population,
            current_generation=generation,
            death_mechanism=config.DEATH_MECHANISM,
            max_population_size=config.MAX_POPULATION_SIZE,
            min_population_size=config.MIN_POPULATION_SIZE,
            max_age=config.MAX_AGE,
        )

        for ind in dead_individuals:
            ind.set_final_generation(generation)
            population.remove(ind)

        for i, coordinate in enumerate(final_coordinates):
            robot = existing_robots[i]
            ind = uuid_to_individual[robot.uuid]
            if ind.final_generation == -1:
                scene.add_robot(robot, pose=Pose(Vector3(coordinate)))

        for ind in population:
            if (
                ind.initial_generation == generation
                and ind.get_robot_uuid() not in existing_robots_uuids
            ):
                random_position = get_random_free_position(
                    config.LIMITS, existing_positions
                )
                scene.add_robot(
                    ind.develop(config.VISUALIZE_MAP), pose=Pose(random_position)
                )
                existing_positions.append(random_position)
        #print(similarity_scores)
    #print("Final similarity scores", similarity_scores)
    #print("Mating: ", mating)
    #print("Meeting: ", meeting)
    #print("Individual count: ", individual_count)
    #print("nov_child population", novelty_child_population)
    #print("nov_child_parents", novelty_child_parents)
    #print("Coordinates per generation", coordinates_per_gen)
    #print("Measures", measures)
    # Example
    #save_dict_to_json(similarity_scores, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 1/similarity.json")
    save_dict_to_json(novelty_child_population, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/run 1/child_population.json")
    save_dict_to_json(novelty_child_parents, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/run 1/child_parents.json")
    save_dict_to_json(coordinates_per_gen, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/run 1/coordinates.json")
    #save_dict_to_json(measures, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_0-045/run 1/measures.json")
    save_dict_to_json(uuid_to_measures, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/run 1/uuid_to_measures.json")
    save_dict_to_json(gen_to_robots, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/morph_065-1/run 1/gen_to_robots.json")
    


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
