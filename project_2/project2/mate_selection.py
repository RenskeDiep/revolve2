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


class MateSelectionStrategy(Enum):
    OPPOSITES = 1
    SIMILAR = 2
    MAX_FITNESS = 3
    MORPHOLOGY = 4

def similarity_score(
    v1, # vector of morphological measures robot 1
    v2, # vector of morphological measures robot 2
    normalize=True,
    method='euclidean'
) -> float:
 
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
        print("TEST")
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
        print(v1, v2)
        sim_score = similarity_score(v1, v2)
        print(sim_score)
        if  sim_score >= similarity_thres_min and sim_score <= similarity_thres_max:
            print("YESSS")
        return sim_score >= similarity_thres_min and sim_score <= similarity_thres_max
        



def reproduce(parent1: Individual, parent2: Individual, rng: np.random.Generator):
    offspring = [Genotype.crossover(parent1.genotype, parent2.genotype, rng)]
    return Individual(genotype=offspring[0], fitness=0.0)
