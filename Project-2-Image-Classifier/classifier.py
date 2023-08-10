#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jon Sweeney
# DATE CREATED: 20230809
# REVISED DATE:
# PURPOSE:  This set of functions to interact with the classifier object
#
##

from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models

def build_classifier(arch):
    """
    Builds the classifier model object

    Returns:
        The classifier model
    """
    # TODO: Add additional Pytorch models 
    print('Building the classifier...')
    if arch == "vgg":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    else:
        print(f"WARNING: {arch} is unsupported. Defaulting to vgg model.")
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(25088, 4096)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(4096, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    model.classifier = classifier
    print('Model is built.')

    return model


def check_validation_set(model, valid_loader, device='cpu'):    
    '''
    Calculate the number correct 

    Returns:
        Percentage correct divided by total
    '''
    correct = 0
    total = 0

    with torch.no_grad():
        for data in valid_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total 


def train_classifier(model, trainloader, validloader, epochs, print_every, learning_rate=0.001, device='cpu'):
    '''
    Trains the classifier
    Parameters:
        model - 
        trainloader - 
        validloader - 
        epochs - 
        print_every -
        criterion -
        optimizer - 
        device - 
    '''
    print('Training the classifier...')
    epochs = epochs
    print_every = print_every
    steps = 0
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_accuracy = check_validation_set(model, validloader, device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss / print_every),
                      "Validation Accuracy: {}".format(round(valid_accuracy, 4)))

                running_loss = 0
    print("Training is Complete")


def save_model(model, data, dir):
    print('Saving the model...')

    checkpoint = {'transfer_model': model.cpu(),
                'input_size': 25088,
                'output_size': 102,
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict()
                }

    torch.save(checkpoint, dir)
    print('Saving is complete.')