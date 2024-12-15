import os
from datetime import datetime
import json
import logging
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

logger = logging.getLogger(__name__)

def select_diverse_samples(embeddings, num_samples):
    """
    Select diverse samples using KMeans clustering.
    embeddings: ndarray of shape (num_samples, embedding_dim)
    num_samples: number of diverse samples to select.
    Returns: indices of selected samples.
    """
    kmeans = KMeans(n_clusters=num_samples, random_state=42)
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    # For each cluster, select the closest sample to the center
    selected_indices = []
    for center in cluster_centers:
        distances = np.linalg.norm(embeddings - center, axis=1)
        selected_indices.append(np.argmin(distances))

    return selected_indices


def calculate_uncertainty(probs):
    """
    Calculate uncertainty using entropy.
    probs: ndarray of shape (num_samples, num_classes)
    Returns: uncertainty scores for each sample.
    """
    return -np.sum(probs * np.log(probs + 1e-9), axis=1)


def save_results(results, args):
    """
    Save the results to a JSON file.

    Args:
        results (dict): The results to save
        output_dir (str): The directory to save the results in
        experiment_name (str): The name of the experiment
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H')
    output_file = f'{args.output_dir}/results_{args.experiment_name}_{timestamp}.json'
    os.makedirs(args.output_dir, exist_ok=True)
    dict_to_save = {
        'args': vars(args),
        'results': results
    }
    with open(output_file, 'w') as f:
        json.dump(dict_to_save, f)
    logger.info(f'Results saved to {output_file}')


def load_results_file(file_path):
    """
    Load results from a JSON file.

    Args:
        file_path (str): The path to the JSON file

    Returns:
        dict: The loaded results
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    results_dict = {}

    # Aggregate the results
    results = pd.DataFrame(data['results']).agg(["mean", "std"], axis=0).to_dict()
    for key, value in results.items():
        for sub_key, sub_value in value.items():
            new_key = f"{key}_{sub_key}"
            results_dict[new_key] = sub_value

    results_dict.update(data['args'])
    return results_dict


def load_all_results(directory):
    """
    Load all results from a directory.

    Args:
        directory (str): The directory to load results from

    Returns:
        list: A list of dictionaries containing the loaded results
    """
    results = []
    for file in os.listdir(directory):
        if file.endswith('.json'):
            results.append(load_results_file(os.path.join(directory, file)))
    return pd.DataFrame(results)