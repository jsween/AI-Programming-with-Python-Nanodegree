import argparse

import json
import torch
from torchvision import datasets, transforms


def get_input_args():
    """
    Retrieves and parses 1 to 7 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's
    argparse module to created and defined these command line arguments. If
    the user fails to provide all of the arguments, then the default values are
    used for the missing arguments.
    Command Line Arguments:
      1. dir : image data directory
      2. save : directory to save checkpoints
      3. arch : Pytorch model
      4. learning_rate : learning rate
      5. hidden_units : number of hidden units
      6. epochs : number of epochs
      7. gpu : use GPU for training
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        help="Required: The path to the directory containing flower images (e.g. flowers/)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="checkpoints/",
        help="The path to the directory to save the checkpoints",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="vgg",
        help="The CNN Model Architecture to use (e.g. vgg)",
    )
    print("Command line arguments parsed")

    return parser.parse_args()


def get_data():
    """
    Gets the data directories
    """
    print("Getting data...")
    data_dir = "flowers"
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    return process_images([train_dir, valid_dir, test_dir])


def process_images(data_dirs):
    train_dir, valid_dir, test_dir = data_dirs
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # Define transforms for training, validation, and testing sets
    modified_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Load datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=modified_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=modified_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=modified_transforms)

    # Define the dataloaders
    trainloaders = torch.utils.data.DataLoader(
        train_datasets, batch_size=32, shuffle=True
    )
    validloaders = torch.utils.data.DataLoader(
        valid_datasets, batch_size=32, shuffle=True
    )
    testloaders = torch.utils.data.DataLoader(
        test_datasets, batch_size=32, shuffle=True
    )

    with open("cat_to_name.json", "r") as f:
        cat_to_name = json.load(f)

    print("Image data loaded")
    return {
        "train": trainloaders,
        "valid": validloaders,
        "test": testloaders,
        "labels": cat_to_name,
    }
