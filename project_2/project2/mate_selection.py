from enum import Enum
import logging
import random
from revolve2.standards.genotype import Genotype
from revolve2.standards.morphological_measures import MorphologicalMeasures
from project2.individual import Individual

import numpy as np
import multineat

from revolve2.experimentation.evolution.abstract_elements import Reproducer
from revolve2.experimentation.rng import make_rng_time_seed
from project2.utils.helpers import similarity_score


class MateSelectionStrategy(Enum):
    OPPOSITES = 1
    SIMILAR = 2
    MAX_FITNESS = 3
    MORPHOLOGY = 4


def mate_decision(
    strategy: MateSelectionStrategy,
    similarity_thres_min: float, 
    similarity_thres_max: float,
    individual1: Individual,
    individual2: Individual,
    population: list[Individual] = None,
    threshold: float = 0.1,
):
    logging.info(
        f"Strategy: {strategy}, individual1: {individual1.fitness}, individual2: {individual2.fitness}"
    )
    if strategy == MateSelectionStrategy.OPPOSITES:
        return abs(individual1.fitness - individual2.fitness) > threshold
    elif strategy == MateSelectionStrategy.SIMILAR:
        return abs(individual1.fitness - individual2.fitness) < threshold
    elif strategy == MateSelectionStrategy.MAX_FITNESS:
        if population is None:
            return (individual1.fitness - individual2.fitness) < 0

        # Calculate the top 50% threshold
        fitness_values = [ind.fitness for ind in population]
        fitness_values.sort(reverse=True)  # Sort in descending order
        top_50_percent_index = len(fitness_values) // 2

        # If population size is odd, we include the middle value in the top 50%
        if len(fitness_values) == 0:
            return False

        top_50_threshold = (
            fitness_values[top_50_percent_index - 1]
            if top_50_percent_index > 0
            else fitness_values[0]
        )

        # Check if individual2's fitness is in the top 50%
        return individual2.fitness >= top_50_threshold
    
    elif strategy == MateSelectionStrategy.MORPHOLOGY:
        sim_score = similarity_score(individual1, individual2)
        print(sim_score)
        if  sim_score >= similarity_thres_min and sim_score <= similarity_thres_max:
            print("YESSS")
        return sim_score >= similarity_thres_min and sim_score <= similarity_thres_max
        



def reproduce(parent1: Individual, parent2: Individual, rng: np.random.Generator):
    offspring = [Genotype.crossover(parent1.genotype, parent2.genotype, rng)]
    return Individual(genotype=offspring[0], fitness=0.0)
