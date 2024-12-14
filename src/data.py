from abc import ABC, abstractmethod
import logging

from datasets import load_dataset, concatenate_datasets
import numpy as np
from src.constants import DatasetName
from src.utils import select_diverse_samples, calculate_uncertainty
import torch

logger = logging.getLogger(__name__)


class Dataset(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess_data(self):
        pass


class Cifar10(Dataset):
    def __init__(self, labeled_ratio):
        self.name = 'uoft-cs/cifar10'
        self.target_variable = 'label'
        self.labeled_ratio = labeled_ratio

        self.load_data()
        self.preprocess_data()

    def load_data(self):
        logger.info('Loading CIFAR10 data...')
        self.data = load_dataset(self.name)
        # TODO: adjust the dataset format here. We can use Torch or Huggignface dataset format
        
    def preprocess_data(self):
        logger.info('Preprocessing CIFAR10 data...')
        # TODO: implement preprocessing
        # TODO: create a new test subset for active learning that keeps only a fraction of the labeled data based on self.labeled_ratio


# TODO: implement other datasets we want to use

class FakeNewsDataset(Dataset):
    def __init__(self, labeled_ratio, processor):
        self.name = 'anson-huang/mirage-news'
        self.target_variable = 'label'
        self.labeled_ratio = labeled_ratio
        self.processor = processor
        self.load_data()
        self.preprocess_data()

    def load_data(self):
        logger.info('Loading Mirage news data...')
        self.data = load_dataset(self.name)

    def transform(self, example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = self.processor([img for img in example_batch["image"] if img.mode == "RGB"], return_tensors='pt')

        # Don't forget to include the labels!
        inputs['label'] = example_batch['label']
        return inputs

    def preprocess_data(self):
        self.data = self.data.with_transform(self.transform)
        self.train = self.data['train']
        self.test = self.data['validation']
        
        # Select labeled_ratio of the data to be labeled
        n_labeled_examples = int(len(self.train) * self.labeled_ratio)
        ids_labeled = range(n_labeled_examples)
        ids_unlabeled = range(n_labeled_examples, len(self.train))
        self.labeled = self.train.select(ids_labeled)
        self.unlabeled = self.train.select(ids_unlabeled)

    def move_samples(self, indices):
        moved_samples = self.unlabeled.select(indices)
        self.unlabeled = self.unlabeled.select(np.setdiff1d(range(len(self.unlabeled)), indices))
        # Concatenate the moved samples to the labeled set
        self.labeled = concatenate_datasets([self.labeled, moved_samples]) # TODO: check if this works

    def select_samples(self, strategy, model, budget):
        if strategy == 'random':
            ids = np.random.randint(0, len(self.unlabeled), budget)
            return ids
        elif strategy in ['uncertainty_diverse', 'uncertainty']:
            model.eval()
            with torch.no_grad():
                # Compute embeddings and class probabilities for unlabeled data
                embeddings = []
                probabilities = []
                unlabeled_loader = torch.utils.data.DataLoader(
                    self.unlabeled, batch_size=32
                )
                for batch in unlabeled_loader:
                    inputs = batch[0].to(model.device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    embeddings.append(model.get_embeddings(inputs).cpu().numpy())  # Replace with your embedding extraction
                    probabilities.append(probs)

                embeddings = np.vstack(embeddings)
                probabilities = np.vstack(probabilities)

            # Calculate uncertainty
            uncertainties = calculate_uncertainty(probabilities)

            if strategy == 'uncertainty':
                query_indices = np.argsort(-uncertainties)[:budget]
            else:
                # Select top uncertain samples
                uncertain_indices = np.argsort(-uncertainties)[:budget*10]
                
                # Refine with diversity measure
                selected_embeddings = embeddings[uncertain_indices]
                diverse_indices = select_diverse_samples(selected_embeddings, budget)
                query_indices = uncertain_indices[diverse_indices]
            return query_indices
        

def get_dataset(name, labeled_ratio=0.1, processor=None):
    """
    Returns a dataset object based on the name provided and creates a new training subset
    that keeps only a fraction of the labeled data.
        
    Args:
        name (str): name of the dataset
        labeled_ratio (float): fraction of labeled data to keep
        
    Returns:
        Dataset: dataset object
    """

    if name == DatasetName.cifar10.value:
        return Cifar10(labeled_ratio)
    elif name == DatasetName.fake_news.value:
        return FakeNewsDataset(labeled_ratio, processor=processor)
    else:
        raise ValueError(f'Dataset {name} not supported. The supported datasets are: {DatasetName.values()}')
