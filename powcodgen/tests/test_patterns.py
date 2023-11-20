import unittest
import torch
import positions
from patterns import get_peak_positions, get_initial_tensors


class TestPatternsFunctions(unittest.TestCase):
    def setUp(self):
        self.ttmin = 4
        self.ttmax = 44
        self.peakrange = 3
        self.datdim = 2048
        self.full_data, self.x, self.plotdata = get_initial_tensors(
                                                    ttmin=self.ttmin,
                                                    ttmax=self.ttmax,
                                                    peakrange=self.peakrange,
                                                    datadim=self.datdim)
        self.crystal_systems = torch.tensor([0, 5])
        self.unit_cells = torch.tensor([ [1, 1, 1, 90, 90, 90],
                                    [4, 5, 6, 89, 91, 92]], dtype=torch.float32)
        self.hkl = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
        self.intensities = torch.tensor([[[1], [1], [1]],
                                        [[2], [4], [6]]], dtype=torch.float32)
        self.matrix, self.valid = positions.get_lattice_matrix(self.unit_cells)
        self.inv_matrix = torch.linalg.inv(self.matrix)
        self.recip_metric_tensor = positions.get_recip_lattice_metric_tensor(self.inv_matrix)

    def test_get_peak_positions(self):
        outputs = get_peak_positions(self.crystal_systems, self.hkl, self.intensities, self.unit_cells)

if __name__ == '__main__':
    unittest.main()