from __future__ import print_function, absolute_import, division
import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
import KratosMultiphysics.KratosUnittest as KratosUnittest
import os

def GetFilePath(fileName):
    return os.path.dirname(os.path.realpath(__file__)) + "/" + fileName

class TestSparseMatrixMultiplication(KratosUnittest.TestCase):
    
    def _sparse_matrix_multiplication(self, type = "saad"):
        space = KratosMultiphysics.UblasSparseSpace()
        
        file_name = "../../../kratos/tests/A.mm"
        
        # Read the matrices
        A = KratosMultiphysics.CompressedMatrix()
        A2 = KratosMultiphysics.CompressedMatrix()
        KratosMultiphysics.ReadMatrixMarketMatrix(GetFilePath(file_name),A)

        try:
            from scipy import sparse, io
            import numpy as np
            missing_scipy = False
        except ImportError as e:
            missing_scipy = True

        if (missing_scipy == False):
            A_python = io.mmread(file_name)
            A_python.toarray()
            
            A2_python = np.dot(A_python, A_python)
            
            # Solve
            if (type == "saad"):
                CSMA.SparseMatrixMultiplicationUtility.MatrixMultiplicationSaad(A, A, A2)
                        
            for i, j in np.nditer(A2_python.nonzero()):
                self.assertAlmostEqual(A2[int(i), int(j)], A2_python[i, j], 1e-3)
        else:
            self.assertTrue(True)
            
    def test_sparse_matrix_multiplication_saad(self):
        self._sparse_matrix_multiplication("saad")
  
if __name__ == '__main__':
    KratosUnittest.main()
