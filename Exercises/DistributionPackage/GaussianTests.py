import unittest

from distributions import Gaussian

class TestGaussianClass(unittest.TestCase):
    def setUp(self):
        self.gaussian = Gaussian(25, 2)
        self.gaussian2 = Gaussian(5, 2)
        self.gaussian3 = Gaussian(10, 1)

    def test_initialization(self): 
        self.assertEqual(self.gaussian.mean, 25, 'incorrect mean')
        self.assertEqual(self.gaussian.stdev, 2, 'incorrect standard deviation')

    def test_pdf(self):
        self.assertEqual(round(self.gaussian.pdf(25), 5), 0.19947,\
         'pdf function does not give expected result') 

    def test_meancalculation(self):
        self.gaussian.read_data_file('TestData/numbers.txt', True)
        self.assertEqual(self.gaussian.calculate_mean(),\
         sum(self.gaussian.data) / float(len(self.gaussian.data)), 'calculated mean not as expected')

    def test_stdevcalculation(self):
        self.gaussian.read_data_file('TestData/numbers.txt', True)
        self.assertEqual(round(self.gaussian.stdev, 2), 92.87, 'sample standard deviation incorrect')
        self.gaussian.read_data_file('TestData/numbers.txt', False)
        self.assertEqual(round(self.gaussian.stdev, 2), 88.55, 'population standard deviation incorrect')
    
    def test_add(self):
        guassian_sum = self.gaussian2 + self.gaussian3
        self.assertEqual(guassian_sum.mean, 15, 'calculated mean after addition not as expected')
        self.assertEqual(round(guassian_sum.stdev, 2), 2.24, 'calculated stdev after addition not as expected')

        gaussian_one = Gaussian(25, 3)
        gaussian_two = Gaussian(30, 4)
        gaussian_sum = gaussian_one + gaussian_two
        self.assertEqual(gaussian_sum.mean, 55, 'calculated mean after addition not as expected')
        self.assertEqual(gaussian_sum.stdev, 5, 'calculated stdev after addition not as expected')
    
    def test_sub(self):
        gaussian_one = Gaussian(100, 10)
        gaussian_two = Gaussian(40, 5)
        gaussian_diff = gaussian_one - gaussian_two
        self.assertEqual(gaussian_diff.mean, 60, 'calculated mean after subtraction not as expected')
        self.assertEqual(round(gaussian_diff.stdev, 2), 11.18, 'calculated stdev after subtraction not as expected')

    def test_repr(self):
        self.assertEqual(str(self.gaussian), 'mean: 25, stand dev: 2', 'incorrect repr string returned')

if __name__ == '__main__':
    unittest.main()