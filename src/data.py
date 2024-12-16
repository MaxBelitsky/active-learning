from abc import ABC, abstractmethod
import logging
import random

from datasets import load_dataset, concatenate_datasets, DatasetDict
import numpy as np
from src.constants import DatasetName, QueryStrategy
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
        
    def preprocess_data(self):
        logger.info('Preprocessing CIFAR10 data...')
        pass


class FakeNewsDataset(Dataset):
    def __init__(self, labeled_ratio, incorrect_labels_ratio, processor):
        self.name = 'anson-huang/mirage-news'
        self.target_variable = 'label'
        self.labeled_ratio = labeled_ratio
        self.incorrect_labels_ratio = incorrect_labels_ratio
        self.processor = processor
        self.load_data()
        self.preprocess_data()

    def load_data(self):
        logger.info('Loading Mirage news data...')
        self.data = load_dataset(self.name)
        splits_to_keep = ['train', 'validation']
        self.data = DatasetDict({split: self.data[split] for split in self.data if split in splits_to_keep})

    def transform(self, example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = self.processor([img for img in example_batch["image"] if img.mode == "RGB"], return_tensors='pt')
        
        # Don't forget to include the labels!
        inputs['label'] = example_batch['label']

        # Add embeddings if they exist
        if 'embeddings' in example_batch:
            inputs['embeddings'] = torch.tensor(example_batch['embeddings'])

        return inputs

    def preprocess_data(self):
        self.data = self.data.filter(lambda x: x['image'].mode == 'RGB')
        self.data = self.data.with_transform(self.transform)
        self.train = self.data['train']
        self.test = self.data['validation']

        # Select labeled_ratio of the data to be labeled
        if self.labeled_ratio == 1:
            self.labeled = self.train
            self.unlabeled = None
        else:
            n_labeled_examples = int(len(self.train) * self.labeled_ratio)
            ids_labeled = range(n_labeled_examples)
            ids_unlabeled = range(n_labeled_examples, len(self.train))
            self.labeled = self.train.select(ids_labeled)
            self.unlabeled = self.train.select(ids_unlabeled)

    def move_samples(self, indices):
        moved_samples = self.unlabeled.select(indices)
        
        # Flip labels if needed
        if self.incorrect_labels_ratio > 0:
            moved_samples = moved_samples.map(lambda example: self.flip_labels(example, flip_ratio=self.incorrect_labels_ratio))

        moved_samples.set_transform(self.transform)
        self.unlabeled = self.unlabeled.select(np.setdiff1d(range(len(self.unlabeled)), indices))
        # Concatenate the moved samples to the labeled set
        self.labeled = concatenate_datasets([self.labeled, moved_samples])
        self.labeled.set_transform(self.transform)


    def flip_labels(self, example, flip_ratio=0.3):
        if random.random() < flip_ratio:  # with `flip_ratio` chance
            example['label'] = 1 - example['label']  # flip the label
        return example

    def select_samples(self, strategy, model, budget):
        if strategy == QueryStrategy.random.value:
            ids = np.random.randint(0, len(self.unlabeled), budget)
            return ids
        elif strategy in [QueryStrategy.uncertainty_diverse.value, QueryStrategy.uncertainty.value]:
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

            if strategy == QueryStrategy.uncertainty.value:
                query_indices = np.argsort(-uncertainties)[:budget]
            else:
                # Select top uncertain samples
                uncertain_indices = np.argsort(-uncertainties)[:budget*10]

                # Refine with diversity measure
                selected_embeddings = embeddings[uncertain_indices]
                diverse_indices = select_diverse_samples(selected_embeddings, budget)
                query_indices = uncertain_indices[diverse_indices]
            return query_indices

    def _extract_features(self, example, model):
        """
        Extract features from the model backbone.

        Args:
            example (dict): The example to extract features from
            model (torch.nn.Module): The model to extract features from
        """
        inputs = example['pixel_values'].to(model.device)
        with torch.no_grad():
            output = model.base_model(inputs)
        example['embeddings'] = output.last_hidden_state
        return example

    def extract_features(self, model, batch_size):
        """
        Extract features from the model backbone for the train and test sets.

        Args:
            model (torch.nn.Module): The model to extract features from
            batch_size (int): The batch size to use for feature extraction
        """
        self.labeled = self.labeled.map(
            self._extract_features,
            fn_kwargs={"model": model},
            batched=True,
            batch_size=batch_size,
        )
        self.test = self.test.map(
            self._extract_features,
            fn_kwargs={"model": model},
            batched=True,
            batch_size=batch_size,
        )
        if self.unlabeled:
            self.unlabeled = self.unlabeled.map(
                self._extract_features,
                fn_kwargs={"model": model},
                batched=True,
                batch_size=batch_size,
            )


def get_dataset(name, labeled_ratio=0.1, incorrect_labels_ratio=0.0, processor=None):
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
        return FakeNewsDataset(labeled_ratio, incorrect_labels_ratio, processor=processor)
    else:
        raise ValueError(f'Dataset {name} not supported. The supported datasets are: {DatasetName.values()}')
