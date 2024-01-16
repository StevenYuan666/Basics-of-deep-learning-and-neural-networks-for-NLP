def train_loop(
    model: nn.Module,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs: int = 3,
    device: str = "cpu",
):
    # TODO: Your code here

    # Move the model to the device
    model.to(device)

    # Return list of f1 scores for each epoch on the validation set
    scores = []

    # Iterate over the epochs
    for epoch in range(n_epochs):
        # Print the epoch number
        print(f"Epoch {epoch + 1}")
        # Set the model to train mode
        model.train()
        # Set train loss to 0
        train_loss = 0
        # Iterate over the training data
        for batch in train_loader():
            # Call forward pass
            y_pred = forward_pass(model, batch, device)
            # Call backward pass
            y_label = torch.tensor(batch["label"], dtype=torch.float).to(device)
            loss = backward_pass(optimizer, y_label, y_pred)
            # Accumulate the loss
            train_loss += loss.item()
        # Evaluate the model
        # Set the model to eval mode
        model.eval()
        # Evaluate the model on the training set
        y_true, y_pred = eval_run(model, train_loader, device)
        # Calculate the loss
        loss = bce_loss(y_true, y_pred)
        # Calculate the f1 score
        f1 = f1_score(y_true, y_pred)
        # Print the loss and f1 score
        print(f"Training loss: {loss:.4f}, F1 score: {f1:.4f}")
        # Evaluate the model on the validation set
        y_true, y_pred = eval_run(model, valid_loader, device)
        # Calculate the loss
        loss = bce_loss(y_true, y_pred)
        # Calculate the f1 score
        f1 = f1_score(y_true, y_pred)
        # Print the loss and f1 score
        print(f"Validation loss: {loss:.4f}, F1 score: {f1:.4f}")
        # Append the f1 score to the list
        scores.append(f1)

    # Return the list of f1 scores
    return scores