import os
import pandas as pd

def merge_csv_files(directory, output_file):
    total_df = pd.DataFrame()

    for folder in os.listdir(directory):
        csv_file = os.path.join(directory, folder, 'action.csv')
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['method'] = folder
            total_df = pd.merge(total_df, df, on=['method', 'action'], how='outer')

    total_df.to_csv(output_file, index=False)

directory = 'data/VOST/sta_res'
output_file = os.path.join( directory, 'merged.csv') 
merge_csv_files(directory, output_file)