import argparse

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
    print('Parsing arguments...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Required: The path to the directory containing flower images (e.g. flowers/)')    
    parser.add_argument('--save', type=str, default='checkpoints/', help='The path to the directory to save the checkpoints')
    parser.add_argument('--arch', type=str, default='vgg', help='The CNN Model Architecture to use (e.g. vgg)')
    print('Command line arguments parsed')

    return parser.parse_args()