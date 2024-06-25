"""
imageSet.py

修改时间: 2024-06-19
作者: Jiaxin Li
功能描述: 根据给定的数据分类以及对应的J_mean\ J_last_mean测分结果,
        绘制不同类别的平均得分的柱状图以及不同action的柱状图,并保存成csv文件

"""

import pandas as pd
import os
import re
import argparse

import matplotlib.pyplot as plt

import re

def extract_verb(sequence):
    """
    Extracts the verb from a given sequence.

    Args:
        sequence (str): The input sequence.

    Returns:
        str or None: The extracted verb if found, None otherwise.
    """
    match = re.search(r'_([a-zA-Z]+)_?', sequence)
    if match:
        return match.group(1)
    else:
        return None
    

def get_split(split_path: str):
    """
    Reads all txt files in the specified path and stores the content of each file as a list in a dictionary.

    Args:
    split_path (str): The path of the txt files.

    Returns:
    dict: A dictionary where the key is a line in the txt file (sequence), 
    and the value is the filename of the txt file (excluding the .txt extension, category).
    """
    sequence_to_category = {}
    for filename in os.listdir(split_path):
        if filename.endswith(".txt"):
            with open(os.path.join(split_path, filename), 'r') as f:
                for line in f.read().splitlines():
                    sequence_to_category[line] = filename[:-4]
    return sequence_to_category


def calculate_average_scores(csv_path: str, seq2cat: dict):
    """
    Calculate the average scores for different categories based on the split_dict.

    Args:
    csv_path (str): The path of the csv file.
    split_dict (dict): A dictionary where the key is the filename of the txt file (excluding the .txt extension), 
    and the value is a list where each element is a line in the txt file.

    Returns:
    dict: A dictionary where the key is the category and the value is a tuple of the average J-Mean and J_last-Mean scores.
    """
    df = pd.read_csv(csv_path)
    category_scores = {}
    


    for index, row in df.iterrows():
        sequence = row['Sequence']
        modified_sequence = re.sub(r'_\d+$', '', sequence)
        action =  extract_verb(modified_sequence)
        
        
        if  modified_sequence in seq2cat.keys():

            category = seq2cat[modified_sequence]
            # print(category ,'---',modified_sequence, ":", action)

            if category not in category_scores.keys():
                category_scores[category] = {'J-Mean': [], 'J_last-Mean': [], 'action_list':set()}
            category_scores[category]['J-Mean'].append(row['J-Mean'])
            category_scores[category]['J_last-Mean'].append(row['J_last-Mean'])


            
            if action not in category_scores[category]['action_list'] :
                category_scores[category][action + '_J-Mean'] = []
                category_scores[category][action + '_J_last-Mean'] = []
                category_scores[category]['action_list'].add(action)
                # print(action)
            category_scores[category][action + '_J-Mean'].append(row['J-Mean'])
            category_scores[category][action + '_J_last-Mean'].append(row['J_last-Mean'])



    for category, scores in category_scores.items():
        category_scores[category]['J-Mean'] = sum(scores['J-Mean']) / len(scores['J-Mean'])
        category_scores[category]['J_last-Mean'] = sum(scores['J_last-Mean']) / len(scores['J_last-Mean'])

        for action in category_scores[category]["action_list"]:
            category_scores[category][action+'_J_last-Mean'] = sum(scores[action+'_J_last-Mean']) / len(scores[action+'_J_last-Mean'])
            category_scores[category][action+'_J-Mean'] = sum(scores[action+'_J-Mean']) / len(scores[action+'_J-Mean'])


    return category_scores


def calculate_average_scores_different_action(csv_path: str):
    """
    Calculates the average scores for different actions based on a CSV file.

    Args:
        csv_path (str): The path to the CSV file containing the scores.

    Returns:
        dict: A dictionary containing the average scores for each action. The keys are the actions
              and the values are dictionaries with keys 'J-Mean' and 'J_last-Mean' representing
              the average scores for each action.
    """
    df = pd.read_csv(csv_path)
    action_scores = {}

    for index, row in df.iterrows():
        sequence = row['Sequence']
        modified_sequence = re.sub(r'_\d+$', '', sequence)
        action = extract_verb(modified_sequence)

        if action not in action_scores.keys():
            action_scores[action] = {'J-Mean': [], 'J_last-Mean': []}
        action_scores[action]['J-Mean'].append(row['J-Mean'])
        action_scores[action]['J_last-Mean'].append(row['J_last-Mean'])

    for action, scores in action_scores.items():
        action_scores[action]['J-Mean'] = sum(scores['J-Mean']) / len(scores['J-Mean'])
        action_scores[action]['J_last-Mean'] = sum(scores['J_last-Mean']) / len(scores['J_last-Mean'])

    return action_scores



def plot_average_action_scores(average_scores, save_dir=None):
    """
    Plot the average scores for different categories.

    Args:
    average_scores (dict): A dictionary where the key is the category and the value is a tuple of the average J-Mean and J_last-Mean scores.
    save_path (str, optional): The path to save the plot. If None, the plot will not be saved. Defaults to None.
    """
    categories = list(average_scores.keys())

    for category in categories:
        action_list = average_scores[category]["action_list"]
        j_mean_scores = [average_scores[category][action+ "_J-Mean"] for action in action_list  ]
        j_last_mean_scores = [average_scores[category][action+'_J_last-Mean'] for action in action_list]
            
        # print(category, ":",action_list)
        x = range(len(action_list))

        plt.figure()  # Create a new figure for each category

        width = 0.4
        plt.bar(x, j_mean_scores, width=width, label='J-Mean', color='g', align='center')
        plt.bar([i + width for i in x], j_last_mean_scores, width=width, label='J_last-Mean', color='r', align='center')

        plt.xlabel('Action')
        plt.ylabel('Average Score')
        plt.title('Average Scores for Different Action in ' + category )
        plt.xticks(x, action_list, rotation='vertical')
        plt.legend()
        plt.tight_layout()

        for i, j_mean_score in enumerate(j_mean_scores):
            plt.text(i, j_mean_score, str(round(j_mean_score, 3)), ha='center', va='bottom')
        for i, j_last_mean_score in enumerate(j_last_mean_scores):
            plt.text(i + width, j_last_mean_score, str(round(j_last_mean_score, 3)), ha='center', va='bottom')

        save_path = os.path.join(save_dir, category + "_action.png")

        if save_path is not None:
            plt.savefig(save_path)

        plt.subplots_adjust(top=0.9)  # Increase the top margin to avoid overlapping text
        plt.show()



def plot_average_scores(average_scores, save_dir=None):
    """
    Plot the average scores for different categories.

    Args:
    average_scores (dict): A dictionary where the key is the category and the value is a tuple of the average J-Mean and J_last-Mean scores.
    save_path (str, optional): The path to save the plot. If None, the plot will not be saved. Defaults to None.
    """
    categories = list(average_scores.keys())
    j_mean_scores = [scores['J-Mean'] for scores in average_scores.values()]
    j_last_mean_scores = [scores['J_last-Mean'] for scores in average_scores.values()]

    x = range(len(categories))


    width = 0.4
    plt.bar(x, j_mean_scores, width=width, label='J-Mean', color='g', align='center')
    plt.bar([i + width for i in x], j_last_mean_scores, width=width, label='J_last-Mean', color='r', align='center')


    plt.xlabel('Category')
    plt.ylabel('Average Score')
    plt.title('Average Scores for Different Categories')
    plt.xticks(x, categories, rotation='vertical')
    plt.legend()
    plt.tight_layout()

    for i, j_mean_score in enumerate(j_mean_scores):
        plt.text(i, j_mean_score, str(round(j_mean_score, 3)), ha='center', va='bottom')
    for i, j_last_mean_score in enumerate(j_last_mean_scores):
        plt.text(i + width, j_last_mean_score, str(round(j_last_mean_score, 3)), ha='center', va='bottom')

    save_path = os.path.join(save_dir, "all_category.png")

    if save_path is not None:
        plt.savefig(save_path)

    plt.subplots_adjust(top=0.9)  # Increase the top margin to avoid overlapping text
    plt.show()

    
def plot_action_scores(average_scores, save_dir=None):
    """
    Plot the average scores for different actionS.

    Args:
    average_scores (dict): A dictionary where the key is the action and the value is a tuple of the average J-Mean and J_last-Mean scores.
    save_path (str, optional): The path to save the plot. If None, the plot will not be saved. Defaults to None.
    """
    categories = list(average_scores.keys())
    j_mean_scores = [scores['J-Mean'] for scores in average_scores.values()]
    j_last_mean_scores = [scores['J_last-Mean'] for scores in average_scores.values()]

    x = range(len(categories))


    width = 0.4
    plt.bar(x, j_mean_scores, width=width, label='J-Mean', color='g', align='center')
    plt.bar([i + width for i in x], j_last_mean_scores, width=width, label='J_last-Mean', color='r', align='center')


    plt.xlabel('Action')
    plt.ylabel('Average Score')
    plt.title('Average Scores for Different Action')
    plt.xticks(x, categories, rotation='vertical')
    plt.legend()
    plt.tight_layout()

    for i, j_mean_score in enumerate(j_mean_scores):
        plt.text(i, j_mean_score, str(round(j_mean_score, 3)), ha='center', va='bottom')
    for i, j_last_mean_score in enumerate(j_last_mean_scores):
        plt.text(i + width, j_last_mean_score, str(round(j_last_mean_score, 3)), ha='center', va='bottom')

    save_path = os.path.join(save_dir, "all_action.png")

    if save_path is not None:
        plt.savefig(save_path)

    plt.subplots_adjust(top=0.9)  # Increase the top margin to avoid overlapping text
    plt.show()


def save_dict_to_csv(data_dict, csv_path, is_action = True):
    """
    Save a dictionary to a CSV file.

    Args:
        data_dict (dict): The dictionary containing the data to be saved.
        csv_path (str): The path to the CSV file.

    Returns:
        None
    """


    if is_action:
    
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        df.reset_index(inplace=True)
        df.columns = ['action', 'J_mean', 'J_last_mean']
    else:
        # print(data_dict)
        data_dict_tmp = {}
        for category in data_dict.keys():
            # if data_dict_tmp not in data_dict_tmp.keys():
            data_dict_tmp[category] = {}
            data_dict_tmp[category]["J_mean"]= data_dict[category]["J-Mean"]
            data_dict_tmp[category]["J_last_mean"]= data_dict[category]["J_last-Mean"]

        # print(data_dict_tmp)
        df = pd.DataFrame.from_dict(data_dict_tmp, orient='index')
        df.reset_index(inplace=True)
        
        df.columns = ['category', 'J_mean', 'J_last_mean']

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description='Calculate average scores for different categories.')
    parser.add_argument('-m', '--method', required=True, help='Method name.')
    parser.add_argument('-d','--dataset', required=True, help='Dataset name.')

    args = parser.parse_args()

    method = args.method
    dataset = args.dataset

    data_path = os.path.join('data', dataset)
    split_path = os.path.join(data_path, 'split')
    res_path = os.path.join(data_path, 'sta_res')
    csv_path = os.path.join(data_path, 'res',f'{method}-per-sequence_results-val.csv')


    if not os.path.exists(split_path):
        raise KeyError(f'Split path {split_path} does not exist.')
    
    # Check if the csv file exists
    if not os.path.exists(csv_path):
        raise KeyError(f'CSV file {csv_path} does not exist.')

    if not os.path.exists(os.path.join(res_path, method)):
        os.mkdir(os.path.join(res_path, method))
        # raise KeyError(f'Split path {split_path} does not exist.')

    split_dict = get_split(split_path)
    average_scores = calculate_average_scores(csv_path, split_dict)

    action_scores = calculate_average_scores_different_action(csv_path)
    # print(action_scores)

    # Plot the average scores for different categories
    plot_average_scores(average_scores=average_scores, save_dir=  os.path.join(res_path, method))
    # Plot the average scores for different actions within each category
    plot_average_action_scores(average_scores=average_scores, save_dir=os.path.join(res_path, method))

    # Plot the average scores for different actions
    plot_action_scores(average_scores=action_scores, save_dir=os.path.join(res_path, method))

    # Save the action scores to a CSV file
    save_dict_to_csv(action_scores, csv_path=os.path.join(res_path, method, "action.csv"), is_action=True)

    # Save the category scores to a CSV file
    save_dict_to_csv(average_scores, csv_path=os.path.join(res_path, method, "category.csv"), is_action=False)






    # Print the average scores
    # print(average_scores)


