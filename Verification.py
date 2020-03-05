import unittest
import numpy as np

class Test(unittest.TestCase):

    def test_eig(self):
        A = np.array([[1,0,0,0],[1,1,0,0],[1,1,1,1],[1,1,1,1]])
        eig = np.linalg.eigvals(A)
        expected = [2.,0.,1.,1.]
        self.assertEqual(list(eig), expected)

if __name__ == '__main__':
    unittest.main()
