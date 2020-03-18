import unittest
import numpy as np
from CGLoc import x_cg, time, fuel_data

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

    def test_cg(self): #tests cg location script with simplified input values
        FMF1 = [] #constant fuel flow
        FMF2 = [] #constant fuel flow

        x_cg_t = x_cg(time, fuel_data, FMF1, FMF2)
        xcg_begin = 0
        xcg_middle = 0
        xcg_end = 0

        self.assertEqual(x_cg_t[0], xcg_begin)
        self.assertEqual(x_cg_t[53430 / 2], xcg_middle)
        self.assertEqual(x_cg_t[-1], xcg_end)

if __name__ == '__main__':
    unittest.main()

