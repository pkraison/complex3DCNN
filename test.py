import unittest
import torch
from model.network import *

class TestComplexConv3d(unittest.TestCase):

    def setUp(self):
        # Set up the model and a sample input
        self.model = ComplexConv3d(3, 5, kernel_size=3, stride=1, padding=1)
        self.sample_input_real = torch.randn(1, 3, 10, 10, 10)
        self.sample_input_imag = torch.randn(1, 3, 10, 10, 10)
        self.sample_input = (self.sample_input_real, self.sample_input_imag)

    def test_forward_pass(self):
        # Check if the forward pass correctly computes the complex convolution
        real_part, imag_part = self.model(self.sample_input)
        self.assertEqual(real_part.shape, torch.Size([1, 5, 10, 10, 10]))
        self.assertEqual(imag_part.shape, torch.Size([1, 5, 10, 10, 10]))

class TestComplexConv3dNet(unittest.TestCase):

    def setUp(self):
        # You need to define `whatever_the_flattened_size_is` and `output_size` appropriately
        self.model = ComplexConv3dNet()
        self.sample_input_real = torch.randn(1, 192, 10, 10, 10)
        self.sample_input_imag = torch.randn(1, 192, 10, 10, 10)
        self.sample_input = (self.sample_input_real, self.sample_input_imag)

    def test_network_output(self):
        # Test the entire network pipeline
        output = self.model(self.sample_input)
        # You need to adjust `expected_output_size` based on your output layer's configuration
        expected_output_size = 100  # Example size
        self.assertEqual(output.shape, torch.Size([1, expected_output_size]))

if __name__ == '__main__':
    unittest.main()
