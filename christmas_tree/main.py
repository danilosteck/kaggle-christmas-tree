from christmas_tree.kaggle_source import *
from christmas_tree.plot_trees import plot_trees
from christmas_tree.genetic_algorithm import GeneticAlgorithm
from christmas_tree.utils import df_to_kaggle_output
from datetime import datetime, timedelta
import pandas as pd

GA = GeneticAlgorithm(
    num_shapes = 2,
    bounds_ranges = [(-100,100), (-100,100), (-180,180)],
    population_size = 200,
    generations = 100,
    mutation_rate = 0.1,
    crossover_rate = 0.9,
    evaluate_func = score_mod,
    print_generations=1
)

st = datetime.now()
result = GA.run()
et = datetime.now()
elapsed_time = et-st
print(f'Tempo total de execução: {elapsed_time.seconds} s')

plot_trees(df_to_kaggle_output(pd.DataFrame(result['best_individual'], columns=['x','y','deg'])))

# Referência para 2 árvores: ~0.75 de Score, sem violações
data = [
    ['002_0','s-0.625','s-.175','s-90.0'],
    ['002_1','s0.202736','s-0.511271','s90.0'],
]
score_mod((pd.DataFrame(data, columns=['id','x','y','deg'])))
plot_trees(pd.DataFrame(data, columns=['id','x','y','deg']))