from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


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
