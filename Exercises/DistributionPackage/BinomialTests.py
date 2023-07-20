import unittest

from distributions import Binomial

class TestBinomialClass(unittest.TestCase):
    def setUp(self):
        self.binomial = Binomial()

    def test_initialization(self):
        self.assertEqual(self.binomial.p, 0.5, 'incorrect default probability')
        self.assertEqual(self.binomial.n, 20, 'incorrect default number of trials')
    
    def test_calculatemean(self):
        self.assertEqual(self.binomial.calculate_mean(), 10, 'incorrect mean')

    def test_stdev(self):
        self.assertEqual(round(self.binomial.calculate_stdev(), 2), 2.24, 'incorrect stdev')

if __name__ == '__main__':
    unittest.main()