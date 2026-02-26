import pandas as pd

def drop_non_informative_columns(df):
    cols_to_drop = [
        'eid',      # id
        'vdate',    #visit date i discharged date ne treba jer imamo lengthofstay        
        'discharged'
    ]
    return df.drop(columns = cols_to_drop)


def encode_categorical_features(df):
    """
    One-hot encoding
    """
    
    #categorical_cols = df.select_dtypes(include = 'object').columns
    categorical_cols = ['rcount','gender','facid']

    df_encoded = pd.get_dummies(
        df,
        columns = categorical_cols,
        drop_first = True
    )

    return df_encoded

def encode_boolean_features(df):
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df
