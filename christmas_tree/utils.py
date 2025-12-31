import pandas as pd
from christmas_tree.kaggle_source import score_mod

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

if __name__ == '__main__':
    data = pd.DataFrame([
                {'x':0.895,'y':-9.354,'deg':-90},
                {'x':0.00987,'y':-0.123,'deg':0},
            ])

    submission = df_to_kaggle_output(data)
    
    df = kaggle_output_to_df(submission)

    score_mod(submission['id'], submission, 'id')

