from christmas_tree.kaggle_source import *
import numpy as np
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
import os

class printError(Exception):
    pass

def plot_trees(df, filename=None):
    plt.figure()  # Create a new figure at the start
    
    new_df = pd.DataFrame()
    for c in ['x','y','deg']:
        if not df[c].str.startswith('s').all():
            new_df[c] = df[c]
        else:
            new_df[c] = df[c].str[1:]

    for _, row  in new_df.iterrows():
        ct = ChristmasTree(row['x'], row['y'], row['deg'])
        plot_polygon(ct.polygon, facecolor="lightblue", edgecolor="blue")
    
    if filename:
        plt.savefig(filename)
        plt.close()  # Close the figure instead of just clearing
    else:
        plt.show()
        plt.close()  # Close after showing

if __name__ == '__main__':
    # df = pd.DataFrame([
    #         {'x':'2','y':'1','deg':'-90'},
    #         {'x':'2','y':'1','deg':'0'},
    #     ])

    # plot_trees(df)

    row_id_column_name = 'id'
    # data = [['002_0', 's-0.5', 's-0.3', 's335'], ['002_1', 's0.49', 's0.21', 's155']]
    data = [
        ['001_0','s-0.0','s-.0','s0.0'],
        ['002_0','s-0.0','s-.0','s90.0'],
        ['002_1','s0.202736','s-0.511271','s90.0'],
        ['003_0','s-0.0','s-.0','s90.0'],
        ['003_1','s0.202736','s-0.511271','s90.0'],
        ['003_2','s-0.8806','s-0.29','s-90.0']
    ]

    # data = pd.read_csv(os.path.join('.','christmas_tree','datasources','sample_submission.csv'))

    submission = pd.DataFrame(columns=['id', 'x', 'y', 'deg'], data=data)
    solution = submission[['id']].copy()
    score(solution, submission, row_id_column_name)
    score_mod(submission)
    plt = plot_trees(submission)
