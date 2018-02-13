from __future__ import print_function, absolute_import, division  # makes KratosMultiphysics backward compatible with python 2.6 and 2.7

import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.ContactStructuralMechanicsApplication as ContactStructuralMechanicsApplication

import KratosMultiphysics.KratosUnittest as KratosUnittest

class DummyClass():
    def DummyMethod(self):
        self.this_assert = True

class TestProcessFactory(KratosUnittest.TestCase):
    def setUp(self):
        pass

    def test_process_factory(self):
        dummy_class = DummyClass()
        process_factory = ContactStructuralMechanicsApplication.ProcessFactoryUtility(dummy_class)
        #dummy_class.DummyMethod()
        process_factory.ExecuteMethod("DummyMethod")
        self.assertTrue(dummy_class.this_assert)

if __name__ == '__main__':
    KratosUnittest.main()
