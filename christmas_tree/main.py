from christmas_tree.kaggle_source import *
from christmas_tree.plot_trees import plot_trees
from christmas_tree.genetic_algorithm import GeneticAlgorithm
from christmas_tree.utils import df_to_kaggle_output
from datetime import datetime, timedelta
import pandas as pd
from christmas_tree.simulated_annealing import simulated_annealing


# Implementação conjunta GA + SA

GA = GeneticAlgorithm(
    num_shapes = 10,
    bounds_ranges = [(-100,100), (-100,100), (-180,180)],
    population_size = 30,
    generations = 70,
    mutation_rate = 0.3,
    crossover_rate = 0.7,
    evaluate_func = score_mod,
    print_generations=1
)

st = datetime.now()
result_ga = GA.run()
et = datetime.now()
elapsed_time = et-st
print(f'Tempo total de execução: {elapsed_time.seconds} s | Melhor indivíduo: {result_ga['best_individual']}')

result_sa = simulated_annealing(
        initial_solution=result_ga['best_individual'], 
        bounds_ranges=[
            (pd.DataFrame(result_ga['best_individual'])[0].min(),pd.DataFrame(result_ga['best_individual'])[0].max()),
            (pd.DataFrame(result_ga['best_individual'])[1].min(),pd.DataFrame(result_ga['best_individual'])[1].max()),
            (-180,180)], 
        evaluate_func=score_mod, 
        iterations=5000, 
        initial_temp=1000, 
        cooling_rate=0.9999
        )

plot_trees(tuple_to_kaggle_output(result_ga['best_individual']))
plot_trees(tuple_to_kaggle_output(result_sa['best_solution']))
