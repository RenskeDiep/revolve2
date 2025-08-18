"""Main script for the example."""

import logging

from project2.config import Config
import multineat
from project2.genotype import Genotype
from project2.individual import Individual

from uuid import UUID


from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from project2.stats import Statistics


from .robot_evolution import ModularRobotEvolution
from .parent_selector import ParentSelector
from .survivor_selector import SurvivorSelector
from .evaluator import Evaluator
from .crossover_reproducer import CrossoverReproducer
import copy

import multineat

from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from project2.utils.helpers import similarity_score
from itertools import combinations

import math


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

def run_standard_setup(
    config: Config, stats_folder: str = "stats/standard_setup"
) -> None:
    """Run the program."""
    # Set up logging.
    setup_logging(file_name="log.txt")
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



    # Set up the random number generator.
    rng = make_rng_time_seed()

    # CPPN innovation databases.
    # If you don't understand CPPN, just know that a single database is shared in the whole evolutionary process.
    # One for body, and one for brain.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    """
    Here we initialize the components used for the evolutionary process.

    - evaluator: Allows us to evaluate a population of modular robots.
    - parent_selector: Allows us to select parents from a population of modular robots.
    - survivor_selector: Allows us to select survivors from a population.
    - crossover_reproducer: Allows us to generate offspring from parents.
    - modular_robot_evolution: The evolutionary process as a object that can be iterated.
    """
    evaluator = Evaluator(
        headless=True,
        num_simulators=config.NUM_SIMULATORS,
        plane_size=config.LIMITS.calculate_plane_size(),
        movement_weight=config.MOVEMENT_WEIGHT,
        fitness_function_algorithm=config.FITNESS_FUNCTION_ALGORITHM,
    )

    # TODO: figure out offspring size, currently it's half the population size
    parent_selector = ParentSelector(
        offspring_size=config.POPULATION_SIZE // 2, rng=rng
    )
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )

    modular_robot_evolution = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        evaluator=evaluator,
        reproducer=crossover_reproducer,
    )

    # Create an initial population as we cant start from nothing.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_fitnesses, initial_all_fitness_metrics = evaluator.evaluate(
        initial_genotypes
    )

    # Create a population of individuals, combining genotype with fitness.
    population = [
        Individual(genotype, fitness, 0, fitness_metrics=all_fitness_metrics)
        for genotype, fitness, all_fitness_metrics in zip(
            initial_genotypes,
            initial_fitnesses,
            initial_all_fitness_metrics,
            strict=True,
        )
    ]
    uuid_to_robot: dict[str, ModularRobot] = {}
    uuid_to_individual: dict[str, Individual] = {}

    for individual in population:
        
        robot = individual.develop(config.VISUALIZE_MAP)
        if individual.get_robot_uuid() is None:
                logging.warning("none found in population")
        #individual.develop()
        uuid_to_robot[individual.get_robot_uuid()] = robot
        uuid_to_individual[str(individual.robot.uuid)] = individual
        #uuid_to_individual[str(individual.get_robot_uuid())] = individual
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

    # Set the current generation to 0.
    generation_index = 0

    stats = Statistics(stats_folder)

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation_index < config.ITERATIONS:
        logging.info(f"Generation {generation_index} / {config.ITERATIONS}.")

        for individual in population:
            if individual.get_robot_uuid() is None:
                logging.warning("none found in population")

        population, children, parents, parent_kwargs = modular_robot_evolution.step(
            population,
            generation_index=generation_index,
            stats=stats,
        )
        i = 0
        for child in children:
            child_individual = Individual(
            genotype=child, fitness=0.0, initial_generation=generation_index
        )
            child_individual.develop(config.VISUALIZE_MAP)
            child_robot = child.develop(config.VISUALIZE_MAP)
            uuid_to_individual[str(child_robot.uuid)] = child
            uuid_to_robot[child_robot.uuid] = child_robot
            child_uuid = child_robot.uuid
            parent1, parent2 = parents[i]
            i +=1
            parent1_ind = parent_kwargs["parent_population"][parent1]
            parent2_ind = parent_kwargs["parent_population"][parent2]
            parent1_uuid = parent1_ind.robot.uuid
            parent2_uuid = parent2_ind.robot.uuid
            stats.increment_offspring_count(parent1_uuid)
            stats.increment_offspring_count(parent2_uuid)
            sim_offspring = uuid_to_individual[str(child_robot.uuid)]
            #similarity of child to parents
            #print("Parent1", parent1_ind)
            #print("Child", child)
            parent_score1 = similarity_score(child_individual, parent1_ind)
            parent_score2 = similarity_score(child_individual, parent2_ind)
            novelty_child_parents[str(child_uuid)].append((parent_score1, parent_score2, generation_index))
        
        for individual in population:
            uuid_to_individual[str(individual.robot.uuid)] = individual
            if individual.robot.uuid not in uuid_to_measures.keys():
                measures1 = MorphologicalMeasures(individual.robot.body)
                v1 = np.array([
                    measures1.num_modules,
                    measures1.num_bricks,
                    measures1.branching,
                    measures1.limbs,
                    measures1.length_of_limbs,
                    measures1.coverage,
                    measures1.symmetry,
                    ], dtype=float)
                uuid_to_measures[str(individual.robot.uuid)].append(v1.tolist())

        seen_pairs = set()
        for child in children:
            #similarity of child to population
            for robot2 in population:
                if robot2.robot.uuid == child_uuid:
                    continue

                pair = tuple(sorted([child_uuid, robot2.robot.uuid]))  # order-independent

                if pair in seen_pairs:
                   continue
                seen_pairs.add(pair)
                #print(population)
                #print(len(population))
                #print(len(uuid_to_individual))
                #print("Individ uuid", uuid_to_individual.keys())
                #print(str(robot2.robot.uuid))
                #print(str(robot2.get_robot_uuid()))
                sim2 = uuid_to_individual[str(robot2.robot.uuid)]
                nov_score = similarity_score(child_individual, sim2)
                novelty_child_population[str(child_uuid)].append((str(robot2.robot.uuid), nov_score, generation_index))

        for robot in population:
            gen_to_robots[generation_index].append(str(robot.robot.uuid))

        stats.flush_to_json(f"generation_{generation_index}")
        save_dict_to_json(novelty_child_population, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/run 1/child_population.json")
        save_dict_to_json(novelty_child_parents, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/run 1/child_parents.json")
        save_dict_to_json(uuid_to_measures, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/run 1/uuid_to_measures.json")
        save_dict_to_json(gen_to_robots, "C:/Users/rensk/Documents/Amsterdam/revolve2/stats/standard_setup_k2/run 1/gen_to_robots.json")
    

        generation_index += 1
