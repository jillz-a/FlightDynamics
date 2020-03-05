import unittest

from MatrixCalculation import *

class Test(unittest.TestCase):

    def test_eig(self):
        A = np.array([[1,0,0,0],[1,1,1,1],[1,0,1,1],[1,0,1,1]])
        eig = np.linalg.eig(A)
        expected =
        self.assertTrue(abs(eig - expected)/expected < 0.03)

if __name__ == '__main__':
    unittest.main()
