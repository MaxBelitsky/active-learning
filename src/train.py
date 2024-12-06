import torch
from src.eval import evaluate

def train(model, optimizer, criterion, dataset, args, epsilon):
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
    num_iterations = args['num_iterations']
    query_strategy = args['query_strategy']
    query_budget = args['query_budget']
    
    val_results = []

    for iteration in range(num_iterations):

        train_loader = torch.utils.data.DataLoader(dataset.labeled, batch_size=args['batch_size'], shuffle=True)
        
        for epoch in range(args['num_epochs']):
            for batch in train_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate the model
        test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=args['batch_size'], shuffle=False)
        results = evaluate(model, test_loader)
        print(f'Iteration {iteration}: {results}')
        val_results.append(results)
        if results['accuracy'] > epsilon:
            break
        
        torch.save(model.state_dict(), f'saves/model_{iteration}.pt')

        query_indices = dataset.select_samples(query_strategy, model, query_budget)
        # Select samples to query from the dataset using the query strategy
        dataset.move_samples(query_indices)

    return model