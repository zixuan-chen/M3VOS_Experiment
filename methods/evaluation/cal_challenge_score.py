import pandas as pd
import numpy as np
import os
import sys


def pause_or_exit():
    print("The program is paused. Press Enter to continue or 'q' to exit.")
    try:
        # Wait for user input; if 'q' is entered, exit the loop and program
        if input().lower() == 'q':
            print("The program has exited.")
            sys.exit()
    except KeyboardInterrupt:
        # If the user interrupts with Ctrl+C, allow exit
        print("The program has exited.")
        sys.exit()

def merge_dataframes(*dataframes):
    # First, convert all DataFrames to a list if they are not already
    if not isinstance(dataframes, list):
        dataframes = list(dataframes)
    
    # Check if at least one DataFrame is provided
    if not dataframes:
        raise ValueError("No dataframes provided for merging.")
    
    # Sort all DataFrames by the 'Sequence' column to ensure the last row for each 'Sequence' comes from the last DataFrame
    sorted_dataframes = [df.sort_values('Sequence') for df in dataframes]
    
    # Merge all DataFrames, with ignore_index=True to reindex
    combined_df = pd.concat(sorted_dataframes, ignore_index=True)
    
    # Remove duplicates based on the 'Sequence' column, keeping the last row from the last DataFrame
    unique_df = combined_df.drop_duplicates(subset='Sequence', keep='last')
    
    return unique_df


def calculate_subset_average(df, subset):
    # Filter the DataFrame to only include rows in the subset
    filtered_df = df[df['Sequence'].isin(subset)]
    
    # Calculate the averages of J-Mean, J_last-Mean, and J_cc-Mean
    j_mean_average = filtered_df['J-Mean'].mean()
    j_last_mean_average = filtered_df['J_last-Mean'].mean()
    j_cc_mean_average = filtered_df['J_cc-Mean'].mean()
    
    return j_mean_average, j_last_mean_average, j_cc_mean_average



def calculate_averages_from_dir(dir_path, csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check if the input DataFrame contains the necessary columns
    if 'Sequence' not in df.columns or 'J-Mean' not in df.columns or 'J_last-Mean' not in df.columns or "J_cc-Mean" not in df.columns:
        raise ValueError("CSV file must contain 'Sequence', 'J-Mean', 'J_last-Mean', and 'J_cc-Mean' columns.")
    
    
    # Initialize an empty list to store the results
    results = []
    
    # Iterate through each file in the directory
    for filename in os.listdir(dir_path):
        if filename == "val.txt":
            continue
        # Construct the full path to the file
        file_path = os.path.join(dir_path, filename)
        
        # Ensure it is a file and not a directory
        if os.path.isfile(file_path):
            # Read the subset from the file, assuming each line contains a Sequence name
            with open(file_path, 'r') as f:
                subset = [line.strip().replace('_id', '') for line in f.readlines()]
            # Calculate the averages of J-Mean, J_last-Mean, and J_cc-Mean for this subset
            j_mean_average, j_last_mean_average, j_cc_mean_average = calculate_subset_average(df, subset)
            
            # Add the results to the list
            results.append({
                'subset': filename,
                'j_mean_average': j_mean_average,
                'j_last_mean_average': j_last_mean_average,
                "J_cc_mean_average": j_cc_mean_average
            })
            
            
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    directory = os.path.dirname(csv_file_path)

    # Construct the new file path
    new_csv_path = os.path.join(directory, "challenge_score.csv")

    # Save the results_df to the new file path
    results_df.to_csv(new_csv_path, index=False)
    
    return results_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets_root', type=str)
    parser.add_argument('--week_num', type=int)
    parser.add_argument('--result_csv_path', type=str)
    args, _ = parser.parse_known_args()

    if "ROVES" in args.datasets_root:
        args.ImageSets_path = os.path.join(args.datasets_root, "ROVES_week_" + str(args.week_num), "ImageSets")
    else:
        args.ImageSets_path = os.path.join(args.datasets_root, "ImageSets")
    

    week_filename = args.result_csv_path
    subset_dir = args.ImageSets_path

    result_df = calculate_averages_from_dir(subset_dir, week_filename)
    
    print(result_df)