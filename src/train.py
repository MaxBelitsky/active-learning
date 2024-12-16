import torch
from src.eval import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import jsonlines


def train_cycle(model, optimizer, scheduler, criterion, dataset, args, iteration, writer=None, train_head_only=False):
    model.train()
    train_loss = []
    train_loader = DataLoader(dataset.labeled, batch_size=args.batch_size, shuffle=True)
    
    for epoch in tqdm(range(args.num_epochs)):
        train_loss_epoch = []
        iter = 0
        for batch in train_loader:
            if train_head_only:
                inputs = batch["embeddings"].to(args.device)
            else:
                inputs = batch["pixel_values"].to(args.device)
            labels = batch["label"].to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if train_head_only:
                logits = outputs[:, 0, :] # Take the first token's output as in ViTForImageClassification implementation
            else:
                logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss_epoch.append(loss.item()) 
            if writer:
                writer.add_scalar("Train Loss", loss.item(), iteration*args.num_epochs + epoch + iter/len(train_loader))
            iter += 1
        train_loss.append(sum(train_loss_epoch)/len(train_loss_epoch))
        print(scheduler.get_last_lr())
        print(f'Epoch {epoch}: {train_loss[-1]}')  
    
    return train_loss 

def train(model, optimizer, scheduler, criterion, dataset, args, epsilon, train_head_only=False):
    """
    Train the model on the dataset. This function should implement the active learning loop.
    the model and dataset ar huggingface 

    Args:
        model (torch.nn.Module): The model to train
        dataset (Dataset): The dataset to train on
        training_args (dict): Training arguments
        active_learning_args (dict): Active learning arguments
        train_head_only (bool): Whether to train only the classifier head

    Returns:
        torch.nn.Module: The trained model
    """

    # Initialize active learning loop
    num_iterations = args.num_iterations
    query_strategy = args.query_strategy
    query_budget = args.query_budget

    # get current date and time
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"{args.log_dir}/{query_strategy}-seed_{args.seed}-{now}")  # Create a SummaryWriter for logging

    val_results = []
    train_loss = []

    # fist train the model on the labeled data
    #train_cycle(model, optimizer, scheduler, criterion, dataset, args, 0, writer=writer, train_head_only=train_head_only)

    if dataset.labeled_ratio == 1:
        # If we have all the data labeled, we can just train the model and return it
        writer.close()
        return model

    # Then do AL
    for iteration in range(1, num_iterations+1):
        query_indices = dataset.select_samples(query_strategy, model, query_budget)
        # Select samples to query from the dataset using the query strategy
        dataset.move_samples(query_indices)
        
        train_loss.extend(
            train_cycle(model, optimizer, scheduler, criterion, dataset, args, iteration, writer=writer, train_head_only=train_head_only)
        )
        
        # Evaluate the model
        test_loader = DataLoader(dataset.test, batch_size=args.batch_size, shuffle=False)
        results = evaluate(model, test_loader, head_only=train_head_only)
        print(f'Iteration {iteration}: {results}')
        val_results.append(results)
        writer.add_scalar("Accuracy", results['accuracy'], iteration)
        writer.add_scalar("Val Loss", results['loss'], iteration)

        if results['accuracy'] > epsilon:
            break
        
        #torch.save(model.state_dict(), f'saves/model_{iteration}.pt') # TODO: set correct path

    with jsonlines.open(f'output/val_results_seed_-{query_strategy}-rat_{str(args.incorrect_labels_ratio).replace(".", "")}-{args.seed}-{now}.jsonl', mode='w') as writer:
        for item in val_results:
            writer.write(item)
    
    writer.close()
    return model