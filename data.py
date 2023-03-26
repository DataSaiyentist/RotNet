### --- MNIST dataset preprocessing ---


import random
import numpy as np

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
from torch.utils.data import TensorDataset, Subset, ConcatDataset


# --- MNIST dataset importation
def load_mnist(cleared = False) :
    """load MNIST dataset from pytorch (if cleared = True, then the image is only in black and white)"""
    
    # Whether the image is only in black and white or not
    if cleared :
        transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307, ), (0.3081, )),
                                        lambda x: x > 0, lambda x: x.float(), ])
    else :
        transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    
    # MNIST dataset loading
    train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transformer) # Train set
    test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transformer) # Test set
    
    return train_data, test_data


# --- Process in (only) black and white MNIST-like dataset
def clear_mnist(data) :
    """in case that we need only black and white images"""
    
    # List of black and white images
    cleared_data = [torch.where(np.abs(data[i][0]) == 1, 0., 1.) for i in range(len(data))]
    
    # Get the list of labels from the original dataset
    labels = [data[i][1] for i in range(len(data))]
    
    # Convert them to tensors
    cleared_data = torch.stack(cleared_data)
    labels = torch.tensor(labels)
    
    return TensorDataset(cleared_data, labels)


# --- Feature extraction from MNIST
def extract_mnist(data) :
    """extract some features from the MNIST dataset"""
    
    # Histogram of projections on x-axis (number of black pixels per column)
    hist_x = [data[i][0][0].sum(dim = 0) for i in range(len(data))]
    hist_x = torch.stack(hist_x)
    
    # Histogram of projections on y-axis (number of black pixels per row)
    hist_y = [data[i][0][0].sum(dim = 1) for i in range(len(data))]
    hist_y = torch.stack(hist_y)
    
    # Left profiles (coordinate of the first transition white/ black starting from the left)
    profiles_left = [data[i][0][0].argmax(axis = 1) for i in range(len(data))]
    profiles_left = torch.stack(profiles_left)
    
    # Right profiles (coordinate of the first transition white/ black starting from the right)
    profiles_right = [torch.flip(data[i][0][0], dims = [1]).argmax(axis = 1) for i in range(len(data))]
    profiles_right = torch.stack(profiles_right)
    
    # Get the list of labels from the original dataset
    labels = [data[i][1] for i in range(len(data))]
    
    # Convert them to tensors 
    extracted_data = torch.cat([hist_x, hist_y, profiles_left, profiles_right], dim = 1)
    labels = torch.tensor(labels)
    
    return TensorDataset(extracted_data, labels)


# --- Reduce the number of samples in MNIST
def reduce_mnist(data, per_label = 10) :
    """take a random subset of 10 * per_label samples from the MNIST dataset"""
    
    reduced_data = []
    
    # Select randomly 10 images per label
    for i in range(10) :
        label_indices = torch.where(data.targets == i)[0]
        subset_indices = random.sample(label_indices.tolist(), per_label)
        reduced_data.append(Subset(data, subset_indices))
            
    return ConcatDataset(reduced_data)


# --- Data augmentation of the reduced MNIST dataset
def extend_mnist(data, per_image = 10) :
    """add new samples thanks to severeal data augmentation techniques"""
    
    # Data augmentation techniques (rotations, translations, scaling and perspective)
    augmentation_transformer = transforms.Compose([transforms.RandomAffine(degrees = 20, translate = (0.1, 0.1), scale = (0.9, 1.1)),
                                                   transforms.RandomPerspective(), transforms.Normalize((0.1307, ), (0.3081, )),
                                                   lambda x: x > 0, lambda x: x.float(), ])
    
    extended_data = []
    extended_labels = []
    for i in range(len(data)) :
        img, label = data[i]
        
        # Add the original image
        extended_data.append(img)
        
        # Generation of per_image new images from the original one
        for _ in range(per_image) :
            img_augmented = augmentation_transformer(img)
            extended_data.append(img_augmented)
            
        # Update the list of labels
        extended_labels.extend([label] * (per_image + 1))
        
    # Convert them to tensors
    extended_data = torch.stack(extended_data)
    extended_labels = torch.tensor(extended_labels)
    
    return TensorDataset(extended_data, extended_labels)


# --- Rotation of the MNIST dataset (by 90°, 180° and 270°)
def rotate_mnist(data) :
    """add new samples thanks to rotations of the MNIST dataset"""
    
    rotated_data = []
    rotated_labels = []
    
    for i in range(len(data)) :
        img, _ = data[i]
        
        # Add the original image
        rotated_data.append(img)
        rotated_labels.append(0)
        
        # Generate 3 rotated images from the original one
        for angle in [90, 180, 270] :
            img_rotated = rotate(img, angle)
            rotated_data.append(img_rotated)
            
        # Update the list of labels
        rotated_labels.extend([90, 180, 270])

    # Convert them to tensors
    rotated_data = torch.stack(rotated_data)
    rotated_labels = torch.tensor(rotated_labels)
    
    return TensorDataset(rotated_data, rotated_labels)