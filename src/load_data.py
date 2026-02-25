import pandas as pd

FILE_NAME = 'data/LengthOfStay.csv'
def load_data(path):
    df = pd.read_csv(path)
    return df