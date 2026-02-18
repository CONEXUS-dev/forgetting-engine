"""
The Forgetting Engine - Core Implementation

A paradigm shift in computational optimization:
Instead of searching for right answers, eliminate wrong answers
while preserving paradoxical contradictions.

Author: Derek Angell
Company: CONEXUS
License: Proprietary - All rights reserved
"""

import random
import numpy as np
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Candidate:
    """Represents a potential solution candidate"""
    solution: Any
    fitness: float
    is_paradox: bool = False
    generation: int = 0
    
    def __post_init__(self):
        if self.fitness is None:
            raise ValueError("Fitness must be provided")


class ForgettingEngine:
    """
    The Forgetting Engine - Strategic elimination with paradox retention
    
    Core innovation: Instead of searching for right answers, eliminate wrong answers
    while keeping a few "weird" contradictions that might lead to breakthroughs.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 elimination_rate: float = 0.35,
                 paradox_rate: float = 0.15,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        """
        Initialize the Forgetting Engine
        
        Args:
            population_size: Number of candidate solutions
            elimination_rate: Fraction of worst solutions to eliminate (0.35 = 35%)
            paradox_rate: Fraction of "weird" solutions to preserve (0.15 = 15%)
            mutation_rate: Probability of random mutations
            crossover_rate: Probability of genetic crossover
        """
        self.population_size = population_size
        self.elimination_rate = elimination_rate
        self.paradox_rate = paradox_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Statistics tracking
        self.generation_count = 0
        self.elimination_count = 0
        self.paradox_count = 0
    
    def strategic_elimination(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Core forgetting mechanism: Remove worst performers while preserving paradoxical options
        
        Args:
            candidates: List of candidate solutions
            
        Returns:
            Surviving candidates after strategic elimination
        """
        # Sort by fitness (higher is better)
        sorted_candidates = sorted(candidates, key=lambda x: x.fitness, reverse=True)
        
        # Calculate how many to keep
        keep_count = int(len(sorted_candidates) * (1 - self.elimination_rate))
        survivors = sorted_candidates[:keep_count]
        
        # Add paradox retention from eliminated candidates
        eliminated = sorted_candidates[keep_count:]
        paradox_candidates = self._select_paradox_options(eliminated)
        survivors.extend(paradox_candidates)
        
        # Update statistics
        self.elimination_count += len(eliminated)
        self.paradox_count += len(paradox_candidates)
        
        return survivors
    
    def _select_paradox_options(self, eliminated: List[Candidate]) -> List[Candidate]:
        """
        Select paradoxical options from eliminated candidates
        
        Paradox candidates are "weird" solutions that might lead to breakthroughs
        despite having poor current fitness.
        
        Args:
            eliminated: List of eliminated candidates
            
        Returns:
            Selected paradox candidates
        """
        if not eliminated:
            return []
        
        # Calculate how many paradox candidates to keep
        paradox_count = max(1, int(len(eliminated) * self.paradox_rate))
        
        # Select candidates with highest "paradox potential"
        # For now, we'll use a simple heuristic: candidates with unusual characteristics
        paradox_candidates = []
        
        # Sort by a combination of factors that might indicate paradox potential
        scored_eliminated = []
        for candidate in eliminated:
            # Simple paradox score: lower fitness but higher diversity
            paradox_score = 1.0 / (candidate.fitness + 1e-6)  # Inverse fitness
            scored_eliminated.append((paradox_score, candidate))
        
        # Sort by paradox score and select top candidates
        scored_eliminated.sort(key=lambda x: x[0], reverse=True)
        
        for score, candidate in scored_eliminated[:paradox_count]:
            candidate.is_paradox = True
            paradox_candidates.append(candidate)
        
        return paradox_candidates
    
    def generate_candidates(self, problem: 'OptimizationProblem', count: int) -> List[Candidate]:
        """
        Generate new candidate solutions
        
        Args:
            problem: The optimization problem to solve
            count: Number of candidates to generate
            
        Returns:
            List of new candidate solutions
        """
        candidates = []
        
        for i in range(count):
            # Generate random solution
            solution = problem.generate_random_solution()
            fitness = problem.evaluate_fitness(solution)
            
            candidate = Candidate(
                solution=solution,
                fitness=fitness,
                generation=self.generation_count
            )
            candidates.append(candidate)
        
        return candidates
    
    def crossover(self, parent1: Candidate, parent2: Candidate, 
                  problem: 'OptimizationProblem') -> Candidate:
        """
        Genetic crossover between two parent candidates
        
        Args:
            parent1: First parent candidate
            parent2: Second parent candidate
            problem: The optimization problem
            
        Returns:
            Offspring candidate
        """
        if random.random() > self.crossover_rate:
            # No crossover, return better parent
            return parent1 if parent1.fitness > parent2.fitness else parent2
        
        # Perform crossover (problem-specific)
        offspring_solution = problem.crossover_solutions(parent1.solution, parent2.solution)
        offspring_fitness = problem.evaluate_fitness(offspring_solution)
        
        return Candidate(
            solution=offspring_solution,
            fitness=offspring_fitness,
            generation=self.generation_count
        )
    
    def mutate(self, candidate: Candidate, problem: 'OptimizationProblem') -> Candidate:
        """
        Apply mutation to a candidate
        
        Args:
            candidate: Candidate to mutate
            problem: The optimization problem
            
        Returns:
            Mutated candidate
        """
        if random.random() > self.mutation_rate:
            return candidate
        
        mutated_solution = problem.mutate_solution(candidate.solution)
        mutated_fitness = problem.evaluate_fitness(mutated_solution)
        
        return Candidate(
            solution=mutated_solution,
            fitness=mutated_fitness,
            generation=self.generation_count
        )
    
    def optimize(self, 
                 problem: 'OptimizationProblem',
                 max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6,
                 verbose: bool = False) -> Candidate:
        """
        Run the Forgetting Engine optimization
        
        Args:
            problem: The optimization problem to solve
            max_iterations: Maximum number of generations
            convergence_threshold: Fitness improvement threshold for convergence
            verbose: Whether to print progress
            
        Returns:
            Best candidate found
        """
        # Initialize population
        population = self.generate_candidates(problem, self.population_size)
        best_candidate = max(population, key=lambda x: x.fitness)
        
        if verbose:
            print(f"Generation 0: Best fitness = {best_candidate.fitness:.6f}")
        
        # Evolution loop
        for generation in range(max_iterations):
            self.generation_count = generation
            
            # Strategic elimination (core forgetting mechanism)
            population = self.strategic_elimination(population)
            
            # Generate new candidates to maintain population size
            while len(population) < self.population_size:
                # Selection tournament
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover and mutation
                offspring = self.crossover(parent1, parent2, problem)
                offspring = self.mutate(offspring, problem)
                
                population.append(offspring)
            
            # Update best candidate
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_candidate.fitness:
                improvement = current_best.fitness - best_candidate.fitness
                best_candidate = current_best
                
                if verbose and generation % 10 == 0:
                    print(f"Generation {generation}: Best fitness = {best_candidate.fitness:.6f} (+{improvement:.6f})")
                
                # Check convergence
                if improvement < convergence_threshold:
                    if verbose:
                        print(f"Converged at generation {generation}")
                    break
        
        if verbose:
            print(f"Final best fitness: {best_candidate.fitness:.6f}")
            print(f"Total eliminations: {self.elimination_count}")
            print(f"Paradox candidates preserved: {self.paradox_count}")
        
        return best_candidate
    
    def _tournament_selection(self, population: List[Candidate], tournament_size: int = 3) -> Candidate:
        """
        Tournament selection for parent selection
        
        Args:
            population: Current population
            tournament_size: Number of candidates in each tournament
            
        Returns:
            Selected candidate
        """
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)


class OptimizationProblem(ABC):
    """
    Abstract base class for optimization problems
    """
    
    @abstractmethod
    def generate_random_solution(self) -> Any:
        """Generate a random solution"""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, solution: Any) -> float:
        """Evaluate the fitness of a solution (higher is better)"""
        pass
    
    @abstractmethod
    def crossover_solutions(self, solution1: Any, solution2: Any) -> Any:
        """Perform crossover between two solutions"""
        pass
    
    @abstractmethod
    def mutate_solution(self, solution: Any) -> Any:
        """Apply mutation to a solution"""
        pass


# Example usage with a simple test problem
class SimpleTestProblem(OptimizationProblem):
    """Simple test problem: maximize sum of values"""
    
    def __init__(self, dimension: int = 10, value_range: Tuple[float, float] = (-10, 10)):
        self.dimension = dimension
        self.value_range = value_range
    
    def generate_random_solution(self) -> np.ndarray:
        return np.random.uniform(self.value_range[0], self.value_range[1], self.dimension)
    
    def evaluate_fitness(self, solution: np.ndarray) -> float:
        # Simple fitness: sum of absolute values
        return np.sum(np.abs(solution))
    
    def crossover_solutions(self, solution1: np.ndarray, solution2: np.ndarray) -> np.ndarray:
        # Simple crossover: average of parents
        return (solution1 + solution2) / 2
    
    def mutate_solution(self, solution: np.ndarray) -> np.ndarray:
        # Simple mutation: add small random noise
        mutation = np.random.normal(0, 0.1, solution.shape)
        return solution + mutation


if __name__ == "__main__":
    # Example usage
    print("The Forgetting Engine - Example Run")
    print("=" * 50)
    
    # Create test problem
    problem = SimpleTestProblem(dimension=20)
    
    # Create forgetting engine
    engine = ForgettingEngine(
        population_size=50,
        elimination_rate=0.35,
        paradox_rate=0.15
    )
    
    # Run optimization
    best_solution = engine.optimize(
        problem=problem,
        max_iterations=100,
        verbose=True
    )
    
    print(f"\nBest solution found: {best_solution.solution}")
    print(f"Best fitness: {best_solution.fitness:.6f}")
    print(f"Solution is paradox: {best_solution.is_paradox}")
