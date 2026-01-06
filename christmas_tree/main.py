from christmas_tree.kaggle_source import *
from christmas_tree.plot_trees import plot_trees
from christmas_tree.genetic_algorithm import GeneticAlgorithm
from christmas_tree.utils import *
from datetime import datetime, timedelta
import pandas as pd
from christmas_tree.simulated_annealing import simulated_annealing
from christmas_tree.swarm_optimization import ParticleSwarmOptimization
import os
import logging

# Configuração para saída em um arquivo
logging.basicConfig(filename=f'results_main_{datetime.now().strftime('%Y%m%d')}.log', level=logging.INFO, format='%(asctime)s; %(message)s')



# # Implementação conjunta GA + SA

# GA = GeneticAlgorithm(
#     num_shapes = 10,
#     bounds_ranges = [(-100,100), (-100,100), (-180,180)],
#     population_size = 30,
#     generations = 70,
#     mutation_rate = 0.3,
#     crossover_rate = 0.7,
#     evaluate_func = score_mod,
#     print_generations=1
# )

# st = datetime.now()
# result_ga = GA.run()
# et = datetime.now()
# elapsed_time = et-st
# print(f'Tempo total de execução: {elapsed_time.seconds} s | Melhor indivíduo: {result_ga['best_individual']}')

# result_sa = simulated_annealing(
#         initial_solution=result_ga['best_individual'], 
#         bounds_ranges=[
#             (pd.DataFrame(result_ga['best_individual'])[0].min(),pd.DataFrame(result_ga['best_individual'])[0].max()),
#             (pd.DataFrame(result_ga['best_individual'])[1].min(),pd.DataFrame(result_ga['best_individual'])[1].max()),
#             (-180,180)], 
#         evaluate_func=score_mod, 
#         iterations=5000, 
#         initial_temp=20000, 
#         cooling_rate=0.99
#         )

# plot_trees(tuple_to_kaggle_output(result_ga['best_individual']))
# plot_trees(tuple_to_kaggle_output(result_sa['best_solution']))


# n_start = 1
# n_end = 10

def GASA(n_start, n_end):
    for i in range(n_start, n_end+1):
        GA = GeneticAlgorithm(
            num_shapes = i,
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
        filename = f'output_GA_{datetime.now().strftime("%Y%m%d_%H%M%S")}_score_{str(result_ga['score']).replace('.', 'p')}.csv'

        tuple_to_kaggle_output(result_ga['best_individual']).to_csv()

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
        
def pso_compact():
    plot_path = os.path.join('christmas_tree','plots', datetime.now().strftime("%Y%m%d"))
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)

    data = []
    total_score = 0
    output_df = pd.DataFrame()
    trt = 0
    for i in range(0,200):
        start_time = datetime.now()
        num_shapes = i+1
        swarm_size = 30*num_shapes
        iterations = min(500,20*num_shapes)
        w, c1, c2 = 0.5, 1.5, 1.5

        pso = ParticleSwarmOptimization(
            num_shapes=num_shapes, 
            bounds_ranges=[(-100,100), (-100,100), (-180,180)], 
            evaluate_func=score_mod,
            population_size=swarm_size,
            iterations=iterations,
            w=0.5,  
            c1=1.5,  
            c2=1.5,  
            print_generations=10 # Print results after each X generations - default 20 generations
        )

        pso_result = pso.run()

        compact_population_result = tuple_to_kaggle_output(initialize_compact_population(num_shapes,.715,.79,1. ))
        compact_result = score_mod(compact_population_result)
        

        if i <= 15 and pso_result['score'] < compact_result['score']:
            consolidated_result = pso_result
            consolidated_result['method'] = 'pso'
            df_consolidated_result = tuple_to_kaggle_output(consolidated_result['best_individual'])
        else:
            consolidated_result = compact_result
            consolidated_result['method'] = 'compact'
            df_consolidated_result = compact_population_result
        
        
        total_score += consolidated_result['score']
        data.append([(i, consolidated_result['score'], consolidated_result['is_valid'], total_score)])
        output_df = pd.concat([output_df, df_consolidated_result], ignore_index=True)
        
        end_time = datetime.now()
        td = int((end_time - start_time).seconds)
        trt += td
        print(f'Num. of shapes: {num_shapes} | iter: {i} | Current score: {consolidated_result['score']} | Method: {consolidated_result['method']} | total score: {total_score} | is valid {consolidated_result['is_valid']} | current runtime: {td}s | total runtime: {trt}s')
        logging.info(f'{i};Swarm Optimization + Compact;{num_shapes};{swarm_size};{iterations};{w};{c1};{c2};{consolidated_result['score']};{consolidated_result['method']};{total_score};{trt}')
        plot_trees(df_consolidated_result, os.path.join(plot_path, f'plt_{num_shapes}_trees_method_{consolidated_result['method']}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_score_{str(consolidated_result['score']).replace('.', 'p')}.png'))
        
    return output_df, total_score, data

if __name__ == '__main__':
    df, score, data  = pso_compact()
