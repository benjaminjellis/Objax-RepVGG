import unittest
from RepVGG import *
import objax


class TestTrainArchitectures(unittest.TestCase):
    """
    Test forward pass of train architectures
    """

    def setUp(self) -> None:
        self.test_in = objax.random.normal((10, 3, 200, 200))

    def test_create_RepVGG_A0(self):
        model = create_RepVGG_A0(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_A1(self):
        model = create_RepVGG_A1(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_A2(self):
        model = create_RepVGG_A2(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B0(self):
        model = create_RepVGG_B0(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B1(self):
        model = create_RepVGG_B1(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B1g2(self):
        model = create_RepVGG_B1g2(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B1g4(self):
        model = create_RepVGG_B1g4(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B2(self):
        model = create_RepVGG_B2(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B2g4(self):
        model = create_RepVGG_B2g4(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B3(self):
        model = create_RepVGG_B3(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B3g2(self):
        model = create_RepVGG_B3g2(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))

    def test_create_RepVGG_B3g4(self):
        model = create_RepVGG_B3g4(deploy = False)
        test_out = model(self.test_in, training = True)
        self.assertEqual(test_out.shape, (10, 1000))
