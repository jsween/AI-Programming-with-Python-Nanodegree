import math 
import matplotlib.pyplot as plt
from .Generaldistribution import Distribution

class Binomial(Distribution):
    """ Binomial distribution class for calculating and 
    visualizing a Binomial distribution.
    
    Attributes:
        mean (float) representing the mean value of the distribution
        stdev (float) representing the standard deviation of the distribution
        data_list (list of floats) a list of floats to be extracted from the data file
        p (float) representing the probability of an event occurring
        n (int) number of trials
            
    """
    def __init__(self, prob=0.5, size=20):

        self.p = prob
        self.n = size

        Distribution.__init__(self, self.calculate_mean(), self.calculate_stdev())

    
    def calculate_mean(self):
        """ Function to calculate the mean from p and n

        Args:
            None

        Returns:
            float: mean of the data set
        """

        self.mean = self.p * self.n
        
        return self.mean
