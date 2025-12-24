import numpy as np
import random
from typing import List, Tuple, Dict, Callable

class GeneticAlgorithm:
    def __init__(self, 
                 num_shapes: int,
                 bounds_ranges: List[Tuple[float, float]],
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 evaluate_func: Callable = None):
        self.num_shapes = num_shapes
        self.bounds_ranges = bounds_ranges  # [(min_x,max_x), (min_y,max_y), (min_deg,max_deg)]
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.evaluate = evaluate_func
        
        # Initialize population: each individual is list of (x,y,deg)
        self.population = self._initialize_population()
        self.best_individual = None
        self.best_score = float('inf')
        
    def _initialize_population(self) -> List[List[Tuple[float, float, float]]]:
        population = []
        for _ in range(self.pop_size):
            individual = []
            for i in range(self.num_shapes):
                x = random.uniform(*self.bounds_ranges[0])
                y = random.uniform(*self.bounds_ranges[1])
                deg = random.uniform(*self.bounds_ranges[2])
                individual.append((x, y, deg))
            population.append(individual)
        return population
    
    def _fitness(self, individual: List[Tuple[float, float, float]]) -> Dict:
        return self.evaluate(individual)
    
    def _selection(self, population: List[List[Tuple[float, float, float]]]) -> List[List[Tuple[float, float, float]]]:
        """Tournament selection."""
        tournament_size = 3
        selected = []
        for _ in range(self.pop_size):
            tournament = random.sample(population, tournament_size)
            winner = min(tournament, key=lambda ind: self._fitness(ind)['score'])
            selected.append(winner)
        return selected
    
    def _crossover(self, parent1: List[Tuple[float, float, float]], 
                   parent2: List[Tuple[float, float, float]]) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]  # Copy to avoid reference issues
        
        # Single point crossover
        point = random.randint(1, self.num_shapes - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def _mutate(self, individual: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        mutated = individual[:]
        for i in range(self.num_shapes):
            if random.random() < self.mutation_rate:
                x_range, y_range, deg_range = self.bounds_ranges
                mutated[i] = (
                    random.uniform(*x_range),
                    random.uniform(*y_range),
                    random.uniform(*deg_range)
                )
        return mutated
    
    def run(self) -> Dict:
        """Run GA optimization."""
        for gen in range(self.generations):
            # Evaluate current population
            fitness_scores = [(i, self._fitness(ind)) for i, ind in enumerate(self.population)]
            
            # Update best
            for idx, result in fitness_scores:
                if result['score'] < self.best_score:
                    self.best_score = result['score']
                    self.best_individual = self.population[idx][:]
            
            # Selection
            selected = self._selection(self.population)
            
            # Generate new population
            new_population = []
            for i in range(0, self.pop_size, 2):
                p1 = selected[i]
                p2 = selected[i + 1] if i + 1 < self.pop_size else selected[0]
                
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < self.pop_size:
                    new_population.append(self._mutate(c2))
            
            self.population = new_population[:self.pop_size]
            
            if gen % 20 == 0:
                print(f"Gen {gen}: Best score = {self.best_score:.2f}")
        
        # Final evaluation of best individual
        best_result = self._fitness(self.best_individual)
        return {
            'best_individual': self.best_individual,
            'best_score': self.best_score,
            **best_result
        }

# Usage example:
# ga = GeneticAlgorithm(num_shapes=3, bounds_ranges=[(0,100),(0,100),(0,360)], evaluate_func=your_evaluate_function)
# result = ga.run()