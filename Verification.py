import unittest
import numpy as np


class Test(unittest.TestCase):

    def test_eig(self): #test eigvals function
        A = np.array([[1,0,0,0],[1,1,0,0],[1,1,1,1],[1,1,1,1]])
        eig = np.linalg.eigvals(A)
        expected = [2.,0.,1.,1.] #calculated analytically
        self.assertEqual(list(eig), expected)

    def test_matmul(self): #tests matmul function
        A = np.array([[2,0],[0,2]])
        B = np.array([[1,1],[1,1]])
        C = np.matmul(A,B)
        expected = np.array([[2,2],[2,2]])
        self.assertEqual(C.all(), expected.all())




if __name__ == '__main__':
    unittest.main()

