import unittest
import torch
from ..intensity import get_MD_PO_components, apply_MD_PO_correction
from ..positions import get_lattice_matrix, get_recip_lattice_metric_tensor, get_d_spacing

class TestIntensityFunctions(unittest.TestCase):
    
    def setUp(self):
        self.hkl = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
        self.unit_cell = torch.tensor([  [1, 1, 1, 90, 90, 90],
                                    [3, 4, 5, 91, 92, 93]], dtype=torch.float32)
        self.matrix, valid = get_lattice_matrix(self.unit_cell)
        self.inv_matrix = torch.linalg.inv(self.matrix)
        self.recip_metric_tensor = get_recip_lattice_metric_tensor(self.inv_matrix)
        self.dspacing = get_d_spacing(self.recip_metric_tensor, self.hkl)

    def test_get_MD_PO_components(self):
        """Test get_MD_PO_components with valid input."""
        factor_std=0.1
        cosP, sinP, factor, PO_axis = get_MD_PO_components(self.hkl,
                                                        self.recip_metric_tensor,
                                                        self.dspacing,
                                                        factor_std=factor_std)
        self.assertEqual(cosP.shape, (2, 3))
        self.assertEqual(sinP.shape, (2, 3))
        self.assertEqual(factor.shape, (2, 1))
        self.assertEqual(PO_axis.shape, (2, 3))

    def test_apply_MD_PO_correction(self):
        """Test the MD PO correction"""
        factor_std=0.1
        intensities = torch.tensor([[1, 1, 1],
                                    [1, 2, 3]], dtype=torch.float32)

        cosP, sinP, factor, PO_axis = get_MD_PO_components(self.hkl,
                                                        self.recip_metric_tensor,
                                                        self.dspacing,
                                                        factor_std=factor_std)
        A_all = (1.0/torch.sqrt(((factor)*cosP)**2+sinP**2/(factor)))**3
        corrected_intensities = apply_MD_PO_correction(intensities, cosP, sinP, factor)

        self.assertTrue(torch.allclose(corrected_intensities, intensities*A_all))

if __name__ == '__main__':
    unittest.main()