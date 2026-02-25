import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# uklanjanje neinformativnih kolona
def drop_non_informative_columns(df):
    cols_to_drop = [
        'eid',      # id
        'vdate',    #`visit date` i `discharged date` ne treba jer imamo lengthofstay        
        'discharged'
    ]
    return df.drop(columns = cols_to_drop)


# obrada kategorickih promenljivih
def encode_categorical_features(df):
    
    #categorical_cols = df.select_dtypes(include = 'object').columns
    categorical_cols = ['rcount','gender','facid']

    df_encoded = pd.get_dummies(
        df,
        columns = categorical_cols,
        drop_first = True
    )

    return df_encoded

# pretvaranje True/False u 1/0
def encode_boolean_features(df):
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df
