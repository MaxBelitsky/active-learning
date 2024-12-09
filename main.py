from argparse import ArgumentParser
import logging

from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from src.data import get_dataset
from src.train import train
from src.eval import evaluate
from src.constants import DatasetName
from src.utils import save_results

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = ArgumentParser()

    # General args
    parser.add_argument('--dataset', type=str, help='Name of the dataset', choices=DatasetName.values())
    parser.add_argument('--model', type=str, default="google/vit-base-patch16-224", help='The model to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--num_iterations', type=int, default=10, help='Number of active learning iterations')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for active learning')

    # Active learning args
    parser.add_argument('--labeled_ratio', type=float, default=0.1, help='Fraction of labeled data to keep')
    parser.add_argument('--n_queries', type=int, default=10, help='Number of queries to make')
    parser.add_argument('--query_strategy', type=str, default='random', help='Query strategy to use')
    parser.add_argument('--query_budget', type=int, default=10, help='Number of samples to query at each iteration')
    # TODO: add more active learning args that we need (e.g. acquisition function, uncertainty measure)

    args = parser.parse_args()
    logger.info("The provided arguments:", args)

    # Load the model
    logger.info(f'Loading model {args.model}')
    model = AutoModelForImageClassification.from_pretrained(args.model)
    processor = AutoImageProcessor.from_pretrained(args.model)

    # Load the dataset
    logger.info(f'Loading dataset {args.dataset} with labeled ratio {args.labeled_ratio}')
    dataset = get_dataset(args.dataset, args.labeled_ratio, processor=processor)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the model
    logger.info('Training the model...')
    model = train(model, optimizer, criterion, dataset, args, epsilon=0.98) # TODO: add more args

    # Evaluate the model
    logger.info('Evaluating the model...')
    results = evaluate(model, dataset.test_data) # TODO: add args for evaluation

    # Save the results
    logger.info('Saving the results...')
    save_results(results, args.output_dir)
