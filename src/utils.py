from datetime import datetime
import json
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

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


def save_results(results, output_dir):
    """
    Save the results to a JSON file.

    Args:
        results (dict): The results to save
        output_dir (str): The directory to save the results in
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file = f'{output_dir}/results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f)
    logger.info(f'Results saved to {output_file}')
