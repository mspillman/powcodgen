import unittest
import torch
from ..positions import get_lattice_matrix, get_recip_lattice_metric_tensor
from ..positions import get_unit_cell_perturbation, get_d_spacing
from ..positions import get_zero_point_error, d_to_tt


class TestPositionsFunctions(unittest.TestCase):

    def setUp(self):
        self.unit_cells = torch.tensor([ [1, 1, 1, 90, 90, 90],
                                    [4, 5, 6, 89, 91, 92]], dtype=torch.float32)
        self.matrix, self.valid = get_lattice_matrix(self.unit_cells)
        self.inv_matrix = torch.linalg.inv(self.matrix)
        self.recip_metric_tensor = get_recip_lattice_metric_tensor(self.inv_matrix)
    
    def test_get_unit_cell_perturbation_valid_input(self):
        """ Check that the unit cell perturbation function is giving sensible
        outputs given the crystal system symmetries.

        cubic = crystal_systems == 0
        hexagonal = crystal_systems == 1
        monoclinic = crystal_systems == 2
        orthorhombic = crystal_systems == 3
        tetragonal = crystal_systems == 4
        triclinic = crystal_systems == 5
        trigonal_h = crystal_systems == 6
        trigonal_r = crystal_systems == 7
        """
        stddev = 0.05
        # check cubic, hexagonal, orthorhombic, tetragonal and trigonal_h cells
        cs_no_angle = torch.tensor([0, 1, 3, 4, 6])
        # check monoclinic, triclinic and trigonal_r cells
        cs_angle = torch.tensor([2, 5, 7])
        perturb_no_angle = get_unit_cell_perturbation(cs_no_angle, stddev=stddev)
        perturb_angle = get_unit_cell_perturbation(cs_angle, stddev=stddev)
        self.assertEqual(torch.abs(perturb_no_angle[:,3:]).sum(), 0.)
        self.assertNotEqual(torch.abs(perturb_angle[:,3:]).sum(), 0.)
        self.assertGreater(perturb_no_angle[:,:3].std(), 0.)
        self.assertGreater(perturb_angle[:,:3].std(), 0.)

    def test_get_lattice_matrix(self):
        """Test lattice matrix with valid inputs."""
        self.assertEqual(torch.diagonal(self.matrix[0]).sum(), 3)
        self.assertTrue(torch.all(self.valid))
        self.assertEqual(self.matrix.shape, (2, 3, 3))

    def test_get_recip_lattice_metric_tensor(self):
        """Test get_recip_lattice_metric_tensor with valid inputs."""
        self.assertTrue(torch.allclose(
                            self.inv_matrix @ self.inv_matrix.permute(0,2,1),
                            self.recip_metric_tensor
                                    )
                        )

    def test_get_d_spacing(self):
        """Test get_d_spacing with valid inputs."""
        hkl = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
        d = get_d_spacing(self.recip_metric_tensor, hkl)
        self.assertTrue(torch.allclose(d[0,0], d[0,1:]))
        self.assertFalse(torch.allclose(d[1,0], d[1,1:]))
        self.assertEqual(d[0,0], 1.)

    def test_d_to_tt(self):
        """Test conversion of d-spacing to twotheta"""
        dspacing = torch.tensor([[[10.], [1.]]])
        wavelength = 1.54056
        tt = d_to_tt(dspacing, wavelength=wavelength)
        self.assertTrue(torch.allclose(tt.squeeze()[0], torch.tensor([8.8355])))
        self.assertTrue(torch.allclose(tt.squeeze()[1], torch.tensor([100.7581])))

    def test_get_zero_point_error(self):
        """Test get_zero_point_error with valid inputs"""
        batchsize = 10
        device = torch.device("cpu")
        dtype = torch.float32
        zpemin = -0.3
        zpemax = 0.3
        zpe = get_zero_point_error(batchsize, device, dtype, zpemin=zpemin, zpemax=zpemax)
        self.assertEqual(zpe.shape, (10, 1))
        self.assertTrue((zpe.min() >= zpemin) and (zpe.max() <= zpemax))

if __name__ == '__main__':
    unittest.main()