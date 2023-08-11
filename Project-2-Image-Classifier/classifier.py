#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Jon Sweeney
# DATE CREATED: 20230809
# REVISED DATE: 20230810
# PURPOSE:  This set of functions to interact with the classifier object
#
##

from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models

from utility import process_image, load_checkpoint


def build_classifier(arch, hidden_units):
    """
    Builds the classifier model object

    Parameters: 
        arch: pytorch model
        hidden_units: number of hidden units
    Returns:
        The classifier model
    """
    print("Building the classifier...")
    if arch == "vgg" or arch == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    elif arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif arch == "resnet" or arch == "resnet152":
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    elif arch == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    elif arch == "squeezenet":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    else:
        print(f"WARNING: {arch} is unsupported. Defaulting to vgg model.")
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(25088, hidden_units)),
                ("relu", nn.ReLU()),
                ("fc2", nn.Linear(hidden_units, 102)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    model.classifier = classifier
    print("Model is built.")

    return model


def check_validation_set(model, valid_loader, device="cpu"):
    """
    Calculate the number correct

    Returns:
        Percentage correct divided by total
    """
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


def train_classifier(
    model, trainloader, validloader, epochs, print_every, learning_rate, device
):
    """
    Trains the classifier

    Parameters:
        model - pytorch model used
        trainloader - training images data
        validloader - validation images data
        epochs - number of epochs
        print_every - print every n rows
        device - cpu or gpu device
    """
    print("Training the classifier...")
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

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_accuracy = check_validation_set(model, validloader, device)
                print(
                    "Epoch: {}/{}... ".format(e + 1, epochs),
                    "Loss: {:.4f}".format(running_loss / print_every),
                    "Validation Accuracy: {}".format(round(valid_accuracy, 4)),
                )

                running_loss = 0
    print("Training is Complete")


def save_model(model, file):
    """
    Saves te classifier model

    Parameters:
        model - pytorch model
        file - file name to save the model to
    """
    print("Saving the model...")

    checkpoint = {
        "transfer_model": model.cpu(),
        "input_size": 25088,
        "output_size": 102,
        "features": model.features,
        "classifier": model.classifier,
        "state_dict": model.state_dict(),
    }

    torch.save(checkpoint, file)
    print(f"Model saved as {file}.")


def predict(image_path, checkpoint, top_k):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Parameters:
        image_path: path to the image data
        checkpoint: saved model to use
        top_k: top k predictions
    Returns:
        The results of the model's prediction
    """

    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model, _ = load_checkpoint(checkpoint)
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(top_k)
        return probs[0].tolist(), classes[0].add(1).tolist()


def display_prediction(image_path, model, cat_to_name):
    """
    Displays the Predictions
    
    Parameters:
        image_path: path to the image data
        model: saved model to use for classification
        cat_to_name: JSON data with category to name mapping
    Returns: 
        None
    """
    probs, classes = predict(image_path, model)
    plant_classes = [cat_to_name[str(cls)] + "({})".format(str(cls)) for cls in classes]
    print(plant_classes, probs)
