
from christmas_tree.kaggle_source import score_mod
from christmas_tree.utils import *
import random
import numpy as np
from typing import List, Tuple, Dict, Callable
import pandas as pd

def simulated_annealing(initial_solution, bounds_ranges, evaluate_func, iterations=500, 
                       initial_temp=1000, cooling_rate=0.95):
    """Simulated Annealing local optimizer."""
    current = initial_solution[:]
    current_score = evaluate_func(tuple_to_kaggle_output(current))['score']
    best = current[:]
    best_score = current_score
    
    temp = initial_temp
    
    for i in range(iterations):
        # Generate neighbor (small perturbations)
        neighbor = perturb_solution(current, bounds_ranges, scale=2.0)
        neighbor_score_dict = evaluate_func(tuple_to_kaggle_output(neighbor))
        neighbor_score = neighbor_score_dict['score']
        
        # Accept if better or probabilistically if worse
        if (neighbor_score_dict['is_valid'] and 
            (neighbor_score < current_score or 
             random.random() < np.exp((current_score - neighbor_score) / temp))):
            
            current = neighbor
            current_score = neighbor_score
            
            if neighbor_score < best_score:
                best = neighbor[:]
                best_score = neighbor_score
        
        temp *= cooling_rate
        
        if i % 100 == 0:
            print(f"SA iter {i}: Score={best_score:.2f}, Temp={temp:.1f}")
    
    return {'best_solution': best, 'best_score': best_score}

def perturb_solution(solution, bounds_ranges, scale=1.0):
    """Small random perturbations to x,y,deg."""
    perturbed = solution[:]
    x_range, y_range, deg_range = bounds_ranges
    
    for i in range(len(solution)):
        x, y, deg = perturbed[i]
        
        # Small Gaussian perturbations
        dx = random.gauss(0, scale)
        dy = random.gauss(0, scale)
        ddeg = random.gauss(0, scale * 2)
        
        new_x = np.clip(x + dx, *x_range)
        new_y = np.clip(y + dy, *y_range)
        new_deg = np.clip((deg + ddeg) % 360, *deg_range)
        
        perturbed[i] = (new_x, new_y, new_deg)
    
    return perturbed

if __name__ == '__main__':
    init_sol = [(np.float64(-0.12048131929076428), np.float64(0.5), np.float64(156.21668832972068)), (np.float64(0.5), np.float64(0.5), np.float64(56.365121330662156))]
    result_sa = simulated_annealing(
        initial_solution=init_sol, 
        bounds_ranges=[(-0.5,0.5),(-0.5,0.5),(-180,180)], 
        evaluate_func=score_mod, 
        iterations=5000, 
        initial_temp=2000, 
        cooling_rate=0.95
        )

    plot_trees(tuple_to_kaggle_output(init_sol))
    plot_trees(tuple_to_kaggle_output(result_sa['best_solution']))
