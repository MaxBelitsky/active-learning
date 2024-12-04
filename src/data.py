from abc import ABC, abstractmethod
import logging

from datasets import load_dataset

from src.constants import DatasetName

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


def get_dataset(name, labeled_ratio=0.1):
    """
    Returns a dataset object based on the name provided and creates a new training subset
    that keeps only a fraction of the labeled data.
        
    Args:
        name (str): name of the dataset
        labeled_ratio (float): fraction of labeled data to keep
        
    Returns:
        Dataset: dataset object
    """
    if name == DatasetName.cifar10:
        return Cifar10(labeled_ratio)
    else:
        raise ValueError(f'Dataset {name} not supported. The supported datasets are: {DatasetName.values()}')
