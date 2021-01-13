from numpy import random
import pandas as pd


def create_data(num_rows, num_x_columns, noise):
    columns = [f'x_{i + 1}' for i in range(num_x_columns)]
    df = pd.DataFrame({
        column: random.normal(1, 100, num_rows)
        for column in columns
    })
    df['y'] = 1
    for i, column in enumerate(columns):
        df['y'] = df['y'] + (df[column] ** 2 % (i + 1))
    df['y'] = df['y'] + 1
    df['y'] = df['y'] * df['y'] / 1e5
    df['y'] += random.normal(0, noise, num_rows)
    return df
