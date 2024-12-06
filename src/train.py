import torch


def train(model, optimizer, criterion, dataset, training_args, active_learning_args):
    """
    Train the model on the dataset. This function should implement the active learning loop.
    the model and dataset ar huggingface 

    Args:
        model (torch.nn.Module): The model to train
        dataset (Dataset): The dataset to train on
        training_args (dict): Training arguments
        active_learning_args (dict): Active learning arguments

    Returns:
        torch.nn.Module: The trained model
    """

    # Initialize active learning loop
    num_iterations = active_learning_args['num_iterations']
    query_strategy = active_learning_args['query_strategy']
    query_budget = active_learning_args['query_budget']
    
    for iteration in range(num_iterations):

        # Train the model on the queried samples
        queried_samples = dataset[query_indices]
        train_dataset = torch.utils.data.ConcatDataset([dataset, queried_samples])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_args['batch_size'], shuffle=True)
        
        for epoch in range(training_args['num_epochs']):
            for batch in train_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        

        # Update the dataset by removing the queried samples
        query_indices = dataset.select_samples(query_strategy, model, query_budget)
        # Select samples to query from the dataset using the query strategy
        dataset.move_samples(query_indices)

    return model