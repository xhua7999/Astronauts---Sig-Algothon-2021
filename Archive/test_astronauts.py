# Unit tests for astronauts.py file
import unittest
import numpy as np
from astronauts import *

class TestisPositionValid(unittest.TestCase):
    def test_day1_price(self):
        price = np.array([50,80,90,70,10000])
        position = np.array([200,90,87,56,1])
        self.assertEqual(isPositionValid(position,price),True)


if __name__ == '__main__':
    unittest.main()