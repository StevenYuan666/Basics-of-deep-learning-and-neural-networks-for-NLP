from typing import Union, Iterable, Callable
import random

import torch.nn as nn
import torch


def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "data/train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "data/validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize(
        text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
        word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words - 1]

    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
        tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


### 1.1 Batching, shuffling, iteration
def build_loader(
        data_dict: dict, batch_size: int = 64, shuffle: bool = False
) -> Callable[[], Iterable[dict]]:
    # TODO: Your code here

    # split the data into premise, hypothesis, and label
    premise = data_dict["premise"]
    hypothesis = data_dict["hypothesis"]
    label = data_dict["label"]

    # create a list of indices
    indices = list(range(len(premise)))

    # shuffle the data if needed
    if shuffle:
        random.shuffle(indices)
        premise = [premise[ix] for ix in indices]
        hypothesis = [hypothesis[ix] for ix in indices]
        label = [label[ix] for ix in indices]

    # Assume each key in data_dict has the same length
    num_batches = len(premise) // batch_size
    remainder = len(premise) % batch_size

    # create a generator that returns a batch of data
    def loader():
        # TODO: Your code here
        count = 0
        while count < num_batches:
            batch = {
                "premise": premise[count * batch_size: (count + 1) * batch_size],
                "hypothesis": hypothesis[count * batch_size: (count + 1) * batch_size],
                "label": torch.tensor(label[count * batch_size: (count + 1) * batch_size], dtype=torch.float)
            }
            count += 1
            yield batch
        # when count == num_batches, return the remaining data
        if remainder != 0:
            batch = {
                "premise": premise[count * batch_size:],
                "hypothesis": hypothesis[count * batch_size:],
                "label": torch.tensor(label[count * batch_size:], dtype=torch.float)
            }
            yield batch

    return loader


### 1.2 Converting a batch into inputs
def convert_to_tensors(text_indices: "list[list[int]]") -> torch.Tensor:
    # TODO: Your code here

    # find the maximum length of the text indices
    max_length = max([len(text) for text in text_indices])

    # convert the text indices into a tensor
    tensor = torch.zeros((len(text_indices), max_length), dtype=torch.long)
    for i, text in enumerate(text_indices):
        tensor[i, :len(text)] = torch.tensor(text)

    return tensor


### 2.1 Design a logistic model with embedding and pooling
def max_pool(x: torch.Tensor) -> torch.Tensor:
    # TODO: Your code here

    # Assume input x is a tensor of shape (batch_size, seq_len, hidden_size)
    # max_pool should return a tensor of shape (batch_size, hidden_size)
    return torch.max(x, dim=1)[0]


class PooledLogisticRegression(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()

        # TODO: Your code here

        # Initialize the embedding layer
        self.embedding = embedding
        # Initialize the logistic regression layer
        self.layer_pred = nn.Linear(2 * self.embedding.embedding_dim, 1)  # (2 * hidden_size, 1)
        # Initialize the sigmoid layer
        self.sigmoid = nn.Sigmoid()


    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()

        # TODO: Your code here

        # Get the embedding of the premise and hypothesis
        premise_emb = emb(premise)  # (batch_size, seq_len, hidden_size)
        hypothesis_emb = emb(hypothesis)  # (batch_size, seq_len, hidden_size)
        # Max pool the premise and hypothesis
        premise_pool = max_pool(premise_emb)  # (batch_size, hidden_size)
        hypothesis_pool = max_pool(hypothesis_emb)  # (batch_size, hidden_size)
        # Concatenate the premise and hypothesis
        concat = torch.cat((premise_pool, hypothesis_pool), dim=1)  # (batch_size, 2 * hidden_size)
        # Get the prediction
        pred = layer_pred(concat)  # (batch_size, 1)
        # Get the probability
        prob = sigmoid(pred)  # (batch_size, 1)
        # Reshape the probability
        prob = prob.view(-1)  # (batch_size, )
        # Return the probability
        return prob



### 2.2 Choose an optimizer and a loss function
def assign_optimizer(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    # TODO: Your code here

    # Initialize the optimizer, we use Adam here, not SGD
    optimizer = torch.optim.Adam(model.parameters(), **kwargs)  # (model.parameters(), lr=3e-4, weight_decay=1e-2)
    return optimizer


def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # TODO: Your code here

    # Don't use nn.BCELoss() here
    # Calculate the loss, average over the batch
    loss = -torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
    return loss


### 2.3 Forward and backward pass
def forward_pass(model: nn.Module, batch: dict, device="cpu"):
    # TODO: Your code here

    # Get the premise, hypothesis, and label
    premise = batch["premise"]  # (batch_size, seq_len)
    hypothesis = batch["hypothesis"] # (batch_size, seq_len)

    # Convert the premise and hypothesis into tensors
    premise = convert_to_tensors(premise).to(device)  # (batch_size, seq_len)
    hypothesis = convert_to_tensors(hypothesis).to(device)  # (batch_size, seq_len)

    # Get the prediction
    y_pred = model(premise, hypothesis)  # (batch_size, )

    # Return the prediction
    return y_pred


def backward_pass(
        optimizer: torch.optim.Optimizer, y: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    # TODO: Your code here

    # Calculate the loss
    loss = bce_loss(y, y_pred)

    # Zero the gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update the parameters
    optimizer.step()

    # Return the loss value
    return loss


### 2.4 Evaluation
def f1_score(y: torch.Tensor, y_pred: torch.Tensor, threshold=0.5) -> torch.Tensor:
    # TODO: Your code here

    # Get the prediction
    y_pred = (y_pred > threshold).float()

    # Calculate the true positive
    tp = torch.sum(y * y_pred)

    # Calculate the false positive
    fp = torch.sum((1 - y) * y_pred)

    # Calculate the false negative
    fn = torch.sum(y * (1 - y_pred))

    # Calculate the precision
    precision = tp / (tp + fp)

    # Calculate the recall
    recall = tp / (tp + fn)

    # Calculate the f1 score
    f1 = 2 * precision * recall / (precision + recall)

    # Return the f1 score
    return f1


### 2.5 Train loop
def eval_run(
        model: nn.Module, loader: Callable[[], Iterable[dict]], device: str = "cpu"
):
    # TODO: Your code here

    # Iterate over the data, predict and collect the labels
    y_true = []
    y_pred = []
    for batch in loader():
        # Get the label
        label = batch["label"].to(device)
        # Get the prediction
        pred = forward_pass(model, batch, device)
        # Collect the label
        y_true.append(label)
        # Collect the prediction
        y_pred.append(pred)

    # Concatenate the labels
    y_true = torch.cat(y_true, dim=0)
    # Concatenate the predictions
    y_pred = torch.cat(y_pred, dim=0)

    # Return the prediction and label
    return y_true, y_pred


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
            y_label = batch["label"].to(device)
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


### 3.1
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int):
        super().__init__()

        # TODO: continue here

        # Initialize the embedding layer
        self.embedding = embedding
        # Initialize the logistic regression layer
        self.layer_pred = nn.Linear(self.embedding.embedding_dim, 1)
        # Initialize the sigmoid layer
        self.sigmoid = nn.Sigmoid()
        # Initialize the activation function
        self.activation = nn.ReLU()
        # Initialize the feedforward layer
        self.ff_layer = nn.Linear(2 * self.embedding.embedding_dim, hidden_size)

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layer(self):
        return self.ff_layer

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layer = self.get_ff_layer()
        act = self.get_activation()

        # TODO: continue here

        # Get the embedding of the premise and hypothesis
        premise_emb = emb(premise)  # (batch_size, seq_len, hidden_size)
        hypothesis_emb = emb(hypothesis)  # (batch_size, seq_len, hidden_size)
        # Max pool the premise and hypothesis
        premise_pool = max_pool(premise_emb)  # (batch_size, hidden_size)
        hypothesis_pool = max_pool(hypothesis_emb)  # (batch_size, hidden_size)
        # Concatenate the premise and hypothesis
        concat = torch.cat((premise_pool, hypothesis_pool), dim=1)  # (batch_size, 2 * hidden_size)
        # Apply the feedforward layer
        ff = ff_layer(concat)  # (batch_size, hidden_size)
        # Apply the activation function
        ff = act(ff)  # (batch_size, hidden_size)
        # Get the prediction
        pred = layer_pred(ff)  # (batch_size, 1)
        # Get the probability
        prob = sigmoid(pred)  # (batch_size, 1)
        # Reshape the probability
        prob = prob.view(-1)  # (batch_size, )
        # Return the probability
        return prob


### 3.2
class DeepNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, num_layers: int = 2):
        super().__init__()

        # TODO: continue here

        # Initialize the embedding layer
        self.embedding = embedding
        # Initialize the logistic regression layer
        self.layer_pred = nn.Linear(2 * self.embedding.embedding_dim, 1)
        # Initialize the sigmoid layer
        self.sigmoid = nn.Sigmoid()
        # Initialize the activation function
        self.activation = nn.ReLU()
        # Initialize the feedforward layers
        self.ff_layers = nn.ModuleList([nn.Linear(2 * self.embedding.embedding_dim, 2 * hidden_size) for _ in range(num_layers)])

    # DO NOT CHANGE THE SECTION BELOW! ###########################
    # # This is to force you to initialize certain things in __init__
    def get_ff_layers(self):
        return self.ff_layers

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    # DO NOT CHANGE THE SECTION ABOVE! ###########################

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layers = self.get_ff_layers()
        act = self.get_activation()

        # TODO: continue here

        # Get the embedding of the premise and hypothesis
        premise_emb = emb(premise)  # (batch_size, seq_len, hidden_size)
        hypothesis_emb = emb(hypothesis)  # (batch_size, seq_len, hidden_size)
        # Max pool the premise and hypothesis
        premise_pool = max_pool(premise_emb)  # (batch_size, hidden_size)
        hypothesis_pool = max_pool(hypothesis_emb)  # (batch_size, hidden_size)
        # Concatenate the premise and hypothesis
        concat = torch.cat((premise_pool, hypothesis_pool), dim=1)  # (batch_size, 2 * hidden_size)
        # Apply the feedforward layers
        for ff_layer in ff_layers:
            concat = ff_layer(concat)
            concat = act(concat)
        # Get the prediction
        pred = layer_pred(concat)  # (batch_size, 1)
        # Get the probability
        prob = sigmoid(pred)  # (batch_size, 1)
        # Reshape the probability
        prob = prob.view(-1)  # (batch_size, )
        # Return the probability
        return prob


if __name__ == "__main__":
    # If you have any code to test or train your model, do it BELOW!

    # Seeds to ensure reproducibility
    random.seed(2022)
    torch.manual_seed(2022)

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Print the device
    print(f"Using device: {device}")

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data")

    train_tokens = {
        "premise": tokenize(train_raw["premise"], max_length=64),
        "hypothesis": tokenize(train_raw["hypothesis"], max_length=64),
    }

    valid_tokens = {
        "premise": tokenize(valid_raw["premise"], max_length=64),
        "hypothesis": tokenize(valid_raw["hypothesis"], max_length=64),
    }

    word_counts = build_word_counts(
        train_tokens["premise"]
        + train_tokens["hypothesis"]
        + valid_tokens["premise"]
        + valid_tokens["hypothesis"]
    )
    index_map = build_index_map(word_counts, max_words=10000)

    train_indices = {
        "label": train_raw["label"],
        "premise": tokens_to_ix(train_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(train_tokens["hypothesis"], index_map)
    }

    valid_indices = {
        "label": valid_raw["label"],
        "premise": tokens_to_ix(valid_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(valid_tokens["hypothesis"], index_map)
    }

    # 1.1
    train_loader = build_loader(train_indices, batch_size=64, shuffle=True)
    valid_loader = build_loader(valid_indices, batch_size=64, shuffle=False)

    # 1.2
    batch = next(train_loader())
    y = batch["label"]

    # 2.1
    embedding = torch.nn.Embedding(10000, 512)  # max_words, embedding_dim
    model = PooledLogisticRegression(embedding)

    # 2.2
    optimizer = assign_optimizer(model, lr=3e-4, weight_decay=1e-2)

    # 2.3
    y_pred = forward_pass(model, batch)
    loss = backward_pass(optimizer, y, y_pred)

    # 2.4
    score = f1_score(y, y_pred)

    # 2.5
    n_epochs = 5


    embedding = torch.nn.Embedding(10000, 512)  # max_words, embedding_dim
    model = PooledLogisticRegression(embedding)
    optimizer = assign_optimizer(model, lr=3e-4, weight_decay=1e-2)

    scores = train_loop(model, train_loader, valid_loader, optimizer, n_epochs, device)

    # 3.1
    n_epochs = 10
    embedding = torch.nn.Embedding(10000, 512)  # max_words, embedding_dim
    model = ShallowNeuralNetwork(embedding, 512)
    optimizer = assign_optimizer(model, lr=3e-4, weight_decay=1e-2)

    scores = train_loop(model, train_loader, valid_loader, optimizer, n_epochs, device)

    # 3.2
    embedding = torch.nn.Embedding(10000, 512)  # max_words, embedding_dim
    model = DeepNeuralNetwork(embedding, 512, num_layers=2)
    optimizer = assign_optimizer(model, lr=3e-4, weight_decay=1e-2)

    scores = train_loop(model, train_loader, valid_loader, optimizer, n_epochs, device)
