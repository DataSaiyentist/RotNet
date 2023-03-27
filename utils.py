### --- Useful functions for the project ---


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

import torch
from torch.utils.data import Subset, DataLoader
import torch.optim as optim


# --- Evaluation of pytorch models
def get_accuracy(y_true, y_pred) :
    """compute the accuracy of a model"""
    
    return int(np.sum(np.equal(y_true, y_pred))) / y_true.shape[0]


# --- Architecture and memory allocation of pytorch models
def get_model_memory(model) :
    """show the memory allocation of a pytorch model and its architecture"""
    
    # Model architecture
    print(model)
    print("Model memory allocation : {:.2e}".format(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)))

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters : {total_params}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training parameters : {total_trainable_params}")


# --- Dataset splitting
def split_train_valid(data, valid_size, random_state = None, is_image = True) :
    """split the dataset into training and validation sets"""
    
    if is_image :
        
        # Take by hand the labels of the dataset
        labels = []
        for i in range(len(data)) :
            _, label = data[i]
            labels.append(label)
        
        # Split indices if the dataset is composed of images
        train_indices, val_indices = train_test_split(list(range(len(labels))), test_size = valid_size, stratify = labels, random_state = random_state)
    
    else :
        # Split indices if the dataset is not composed of images
        train_indices, val_indices = train_test_split(list(range(len(data[:][1]))), test_size = valid_size, stratify = data[:][1], random_state = random_state)
    
    return Subset(data, train_indices), Subset(data, val_indices)


# --- Dataset loading
def load_data(data, batch_size, shuffle = False, num_workers = 2) :
    """load the dataset into a dataloader"""
    
    return DataLoader(data, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    

# --- Training of pytorch models
def train_model(train_loader, val_loader, model = None, output_fn = None, epochs:int = None, optimizer = None, criterion = None, device = None) :
    """train a pytorch model and compute metrics such as loss and accuracy at each epoch"""

    loss_valid, acc_valid = [], []
    loss_train, acc_train = [], []

    for epoch in tqdm(range(epochs)) :

        # --- Training
        model.train()
        running_loss = 0.0
        for _, batch in enumerate(train_loader) :

            # Train on GPU
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Set to zero the gradients
            optimizer.zero_grad()

            # Forward, backward and optimization
            out = model(x = inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        # Compute loss and accuracy after an epoch on the validation set
        model.eval()  
        with torch.no_grad() : # /!\ since we're not training, we don't need to compute the gradients for our outputs and save them
            idx = 0
            
            for batch in val_loader :
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if idx==0 :
                    t_out = model(x = inputs)
                    t_loss = criterion(t_out, labels).view(1).item()
                    t_out = output_fn(t_out).detach().cpu().numpy()
                    t_out = t_out.argmax(axis = 1)  # The class with the highest energy is what we choose as prediction                       
                    ground_truth = labels.detach().cpu().numpy()
                    
                else :
                    out = model(x = inputs)
                    t_loss = np.hstack((t_loss, criterion(out, labels).item())) 
                    t_out = np.hstack((t_out, output_fn(out).argmax(axis = 1).detach().cpu().numpy()))
                    ground_truth = np.hstack((ground_truth, labels.detach().cpu().numpy()))
                idx += 1

            acc_valid.append(get_accuracy(ground_truth, t_out))
            loss_valid.append(np.mean(t_loss))

        # Compute loss and accuracy after an epoch on the training set
        with torch.no_grad() :
            idx = 0
            
            for batch in train_loader :
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if idx==0 :
                    t_out = model(x = inputs)
                    t_loss = criterion(t_out, labels).view(1).item()
                    t_out = output_fn(t_out).detach().cpu().numpy()
                    t_out = t_out.argmax(axis = 1)
                    ground_truth = labels.detach().cpu().numpy()
                    
                else :
                    out = model(x = inputs)
                    t_loss = np.hstack((t_loss, criterion(out, labels).item()))
                    t_out = np.hstack((t_out, output_fn(out).argmax(axis = 1).detach().cpu().numpy()))   
                    ground_truth = np.hstack((ground_truth, labels.detach().cpu().numpy()))
                idx += 1

        acc_train.append(get_accuracy(ground_truth, t_out))
        loss_train.append(np.mean(t_loss))

        print('| Epoch: {}/{} | Train: Loss {:.4f} Accuracy : {:.4f} '\
        '| Val: Loss {:.4f} Accuracy : {:.4f}\n'.format(epoch + 1, epochs, loss_train[epoch], acc_train[epoch], loss_valid[epoch], acc_valid[epoch]))

    return model, (loss_train, acc_train, loss_valid, acc_valid)


# --- Pytorch model backup
def save_model(model, path) :
    """save the model in a .pth file"""
    
    return torch.save(model.state_dict(), path)


# --- Accuracy and loss plots
def plot_accuracy(epochs, loss_train, loss_valid, acc_train, acc_valid) :
    """plot the accuracy and loss functions (for each epoch)"""
    
    fig = plt.figure(figsize = (16, 8))
    
    # --- Metrics plot
    def plot_metric(epochs, metric_train, metric_valid, metric_name) :
        """plot metrics of both datasets"""

        plt.plot(range(1, epochs + 1), metric_train, label = f"Training {metric_name.lower()}")
        plt.plot(range(1, epochs + 1), metric_valid, label = f"Validation {metric_name.lower()}")
        plt.xlabel("epochs")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} functions")
        plt.legend()

    # Plot loss functions
    ax = fig.add_subplot(121)
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
    plot_metric(epochs, loss_train, loss_valid, "Loss")

    # Plot accuracy functions
    ax = fig.add_subplot(122)
    for side in ['right', 'top']:
        ax.spines[side].set_visible(False)
    plot_metric(epochs, acc_train, acc_valid, "Accuracy")
    
    
# --- Pytoch model evaluation
def evaluate_model(model, test_loader, device, num_classes = 10) :
    """evaluate the model on the test set"""
    
    # Evaluate the model on the testloader
    y_true, y_pred = [], []
    with torch.no_grad() :
        
        for inputs, labels in test_loader :
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Show classification report
    target_names = [f"Class {str(i)}" for i in range(num_classes)]
    print("Classification report :")
    print(classification_report(y_true, y_pred, target_names = target_names))

    # Show confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
    plt.subplots(figsize = (10, 10))
    sns.heatmap(cm, annot = True, fmt = '.2f', cmap = 'plasma', square = True,
                xticklabels = target_names, yticklabels = target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')
    plt.show()