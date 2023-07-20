import unittest

from distributions import Binomial

class TestBinomialClass(unittest.TestCase):
    def setUp(self):
        self.binomial = Binomial()

    def test_initialization(self):
        self.assertEqual(self.binomial.p, 0.5, 'incorrect default probability')
        self.assertEqual(self.binomial.n, 20, 'incorrect default number of trials')

if __name__ == '__main__':
    unittest.main()