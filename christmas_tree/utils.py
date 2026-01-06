import pandas as pd
from christmas_tree.kaggle_source import score_mod
import math
import random
from typing import List, Tuple

def tuple_to_kaggle_output(tup:tuple) -> pd.DataFrame:
    return df_to_kaggle_output(pd.DataFrame(tup, columns = ['x','y','deg']))

def df_to_kaggle_output(df:pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    if 'id' not in df.columns:
        new_df['id'] = None
        df_length = len(new_df)
        length_prefix = (str(df_length).zfill(3))
        for n, _ in new_df.iterrows():
            new_df.loc[n, "id"] = str(length_prefix) + '_' + str(n)
    
    data_cols = ['x', 'y', 'deg']
    for c in data_cols:
        new_df[c] = 's' + df[c].astype(str)
    
    return new_df.reindex(columns=['id','x','y','deg'])

def kaggle_output_to_df(df:pd.DataFrame) -> pd.DataFrame:
    data_cols = ['x', 'y', 'deg']
    new_df = df.copy()

    for c in data_cols:
        if not df[c].str.startswith('s').all():
            raise ParticipantVisibleError(f'Value(s) in column {c} found without `s` prefix.')
        new_df[c] = df[c].str[1:].astype(float)

    return new_df.reindex(columns=['id','x','y','deg'])

def initialize_compact_population(num_trees: int, 
                         x_sep: float = 1, 
                         y_sep: float = 1,
                         scale: float = 1) -> List[List[Tuple[float, float, float]]]:

    # Calculate grid dimensions
    grid_side = int(math.sqrt(num_trees))
    extra_trees = num_trees - (grid_side * grid_side)
    
    x_sep_used = x_sep*scale
    y_sep_used = y_sep*scale
    
    base_grid = []

    x_pos = 0
    y_pos = 0
    deg_pos = 0
    y_offset = 0
    
    for i in range(0, num_trees):
        is_new_line = 1 if (i % grid_side) == 0 else 0
        if i > 0:
            if (int(i/grid_side) % 2) == 0:
                deg_pos = 0
                y_offset += -0.69*y_sep_used*is_new_line
            else:
                deg_pos = -180
        
        x_pos = ((i % grid_side) * x_sep_used) if (int(i/grid_side) % 2) == 0 else ((i % grid_side) * x_sep_used) + x_sep_used/2
        y_pos = (int(i/grid_side))* y_sep_used + y_offset
        base_grid.append([x_pos, y_pos, deg_pos])

    return base_grid




if __name__ == '__main__':

    data = []
    total_score = 0
    output_df = pd.DataFrame()
    for i in range(0,200):
        compact_population_result = tuple_to_kaggle_output(initialize_compact_population(i+1,.715,.79,1. ))
        res = score_mod(compact_population_result)
        total_score += res['score']
        data.append([(i, res['score'], res['is_valid'], total_score)])
        output_df = pd.concat([output_df, compact_population_result], ignore_index=True)
        print(f'iter: {i} | Current score: {score_mod(compact_population_result)['score']} | total score: {total_score} | is valid {score_mod(compact_population_result)['is_valid']}')

    plot_trees(tuple_to_kaggle_output(initialize_compact_population(100,.715,.79,1. )))
    score_mod(tuple_to_kaggle_output(initialize_compact_population(10,.715,.79,1. )))

