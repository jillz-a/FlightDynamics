import unittest
import numpy as np
from CGLoc import x_cg, time, fuel_data, flow_eng2, flow_eng1, x_lemac, MAC

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
        x_cg_t, m_fuel_t, FMF = x_cg(time, fuel_data, flow_eng1, flow_eng2)
        xcg_begin = (7.13486 - x_lemac)/MAC
        xcg_shift = (0.055118)/MAC


        self.assertEqual(round(x_cg_t[0],3), round(xcg_begin,3))
        self.assertEqual(round(x_cg_t[34078] - x_cg_t[34082], 3), round(xcg_shift, 3))


if __name__ == '__main__':
    unittest.main()

