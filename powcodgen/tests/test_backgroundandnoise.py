import torch
import unittest
from backgroundnoise import get_noise, get_background

class TestBackgroundNoiseFunctions(unittest.TestCase):

    def test_get_noise_valid_input(self):
        """Test the get_noise function with valid input."""
        calculated_patterns = torch.randn(5, 10)
        noise = get_noise(calculated_patterns)
        self.assertEqual(noise.shape, calculated_patterns.shape)
        self.assertIsInstance(noise, torch.Tensor)

    def test_get_background_valid_input(self):
        """Test the get_background function with valid input."""
        batchsize = 5
        data = torch.linspace(0, 1, steps=10)
        bg = get_background(batchsize, data)
        self.assertEqual(bg.shape, (batchsize, data.numel()))
        self.assertIsInstance(bg, torch.Tensor)

    def test_get_background_parameter_ranges(self):
        """Test the effect of bg_prm_max, bg_prm_min, and degree on get_background function."""
        batchsize = 5
        data = torch.linspace(0, 1, steps=10)
        bg_prm_max, bg_prm_min, degree = 0.03, 0.01, 10
        bg = get_background(batchsize, data, bg_prm_max, bg_prm_min, degree)
        self.assertEqual(bg.shape, (batchsize, data.numel()))
        self.assertIsInstance(bg, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
