#============================================================================
# A python library that provides the functionality and implementation of
# an artificial neural network for classification purposes
#============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
import copy

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after
    a given patience.

    Args:
        patience (int): The number of epochs the training loss has not
        improved
        min_delta (float): The minimum between train/test loss
    """
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > (self.best_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class EmbeddingsDataset(Dataset):
    """Provides a fundamental functionality for the implementation of a
    simple Dataset for word embeddings data. Each item is a single (feature, target)
    pair as a given index.

    Args:


    """
    def __init__(self, X, y, device):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        if not torch.is_tensor(y):
            y= torch.tensor(y, dtype=torch.long)

        self.X = X.to(device)
        self.y = y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    """ Class implementation of a simple mutli-layer ANN with
        dynamic hidden layer input.

        Args:
            input_dim (int): Size of the input features.
            hidden_layers (list or int): List containing the size of each hidden layer,
            dropout_prob(float): Probability of dropping out neurons after each hidden layer.
    """

    def __init__(self, input_dim, hidden_layers, dropout_prob=0.2):
        super(MLP, self).__init__()
        layers = []
        in_features = input_dim

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            in_features = hidden_size

        #Output layer
        layers.append(nn.Linear(in_features, 3))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    epochs,
    patience=5):
    """Function that performs the training of the neural network

    Args:
        model (nn.Module): The instance of the neural network model to be trained.
        train_loader (DataLoader): The DataLoader object tha provides the batches
          of the X and the y.
        test_loader (DataLoader): The DataLoader object that provides the batches
          of the testing X and y.
        criterion (callable): The optimization criterion.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        epochs (int): The number of training epochs to be performed.
        patience(int): The number of epochs to run before early stopping.

    Returns:
        model (torch.nn.Module): the trained model
        train_loss (List[float]): Training loss for each epoch.
        train_acc(List[float]): Training model accuracy
        test_loss (List[float]): Test loss for each epoch.
        test_acc (List[float]): Testing accuracy for each epoch.
        test_predictions (List[float]): All the test predictions.
    """
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    test_predictions = []

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)
    # Initialize best model
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        # Training phase
        # Training mode
        model.train()
        # Accumulators Initialization
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        # Loop through batches of the train loader
        for X_batch, Y_batch in train_loader:
            #Set gradient vector to zero
            optimizer.zero_grad()

            preds = model(X_batch)

            # Loss computation
            loss = criterion(preds, Y_batch)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

            # Number of correct prediction
            _, predicted = torch.max(preds, dim=1)
            correct_preds += (predicted == Y_batch).sum().item()
            total_samples += Y_batch.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        epoch_train_acc = correct_preds / total_samples

        # Validation
        avg_test_loss, avg_test_acc, all_test_predictions = test_model(model, test_loader, criterion)

        train_loss.append(avg_train_loss)
        train_acc.append(epoch_train_acc)
        test_loss.append(avg_test_loss)
        test_acc.append(avg_test_acc)
        test_predictions.append(all_test_predictions)

        print(
            f"Epoch [{epoch+1}/{epochs}], \n",
            f"Train Loss: {avg_train_loss:>10.6f} |",
            f"Train Acc:  {epoch_train_acc:>10.6f}\n",
            f"Test Loss:  {avg_test_loss:>10.6f} | ",
            f"Test Acc:   {avg_test_acc:>10.6f}"
        )

        # Save best model when test loss improves
        if early_stopping.best_loss is None or avg_test_loss < early_stopping.best_loss:
            best_model_state = copy.deepcopy(model.state_dict())
            print("Model improved.!")

        # Check early stopping
        early_stopping(avg_test_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

        # Restore best weights
        model.load_state_dict(best_model_state)
    return model, train_loss, train_acc, test_loss, test_acc, test_predictions



def test_model(model, test_loader, criterion):
    """Function that performs the testing of a
    neural network model.

    Args:
        model(nn.Module): The instance of the trained neural network model
        to be evaluated
        test_loader (DataLoader): The DataLoader object that provide the batches.
        of testing feature vectors along with the corresponding target.
        criterion (callable): The optimization criterion.

    Returns:
        val_loss(float): The average loss generated from validation
        val_acc (float): The average test accuracy of the classification
        val_predictions (List[float]): All the model prediction generated
        during training.
    """

    # Initiate Testing
    model.eval()

    # # Init Lists
    all_test_predictions = []
    val_loss = 0.0
    correct_val_preds = 0
    val_samples = 0

    # Testing
    with torch.no_grad():
       # Loop over batches from the DataLoader
       for X_batch, Y_batch in test_loader:
           # Test Preds
           preds = model(X_batch)

           # Compute the current test loss
           loss = criterion(preds, Y_batch)

           # Accumulate the loss
           val_loss += loss.item() * X_batch.size(0)

           # Compute Accuracy
           _, predicted = torch.max(preds, dim=1)
           correct_val_preds += (predicted == Y_batch).sum().item()
           all_test_predictions.extend(predicted.cpu().tolist())
           val_samples += Y_batch.size(0)


    test_loss = val_loss / len(test_loader.dataset)
    test_acc = correct_val_preds / val_samples

    return test_loss, test_acc, all_test_predictions



def get_predictions(model, data_loader):
    """Function that returns all model predictions and
    respective target values.

    Args:
        model(nn.Module): The trained PyTorch model.
        test_loader (DataLoader): The DataLoader object that provides the batches.
                                  dataset split.

    Returns:
        all_preds (np.ndarray): All the model prediction generated during training.
        all_targets(np.ndarray): Numpy array storing the ground-truth targets.
    """

    # Initiate Testing
    model.eval()

    # # Init Lists
    all_preds = []
    all_targets = []

    # Testing
    with torch.no_grad():
       # Loop over batches from the DataLoader
       for X_batch, Y_batch in data_loader:
           # Test Preds
           preds = model(X_batch)

           # Compute Accuracy
           _, predicted = torch.max(preds, dim=1)
           all_preds.extend(predicted.cpu().numpy())
           all_targets.extend(Y_batch.cpu().numpy())

    return all_preds, all_targets
