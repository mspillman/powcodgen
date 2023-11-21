import unittest
import torch
import positions
from patterns import get_peak_positions, get_initial_tensors, get_PO_intensities
from patterns import get_peak_shape_params, calculate_peaks, combine_peaks_for_full_patterns


class TestPatternsFunctions(unittest.TestCase):

    def setUp(self):
        self.ttmin = 4
        self.ttmax = 44
        self.peakrange = 3
        self.datadim = 2048
        self.full_data, self.x, self.plotdata = get_initial_tensors(
                                                    ttmin=self.ttmin,
                                                    ttmax=self.ttmax,
                                                    peakrange=self.peakrange,
                                                    datadim=self.datadim)
        self.crystal_systems = torch.tensor([0, 5])
        self.unit_cells = torch.tensor([[1, 1, 1, 90, 90, 90],
                                    [4, 5, 6, 89, 91, 92]], dtype=torch.float32)
        self.hkl = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float32)
        self.intensities = torch.tensor([[1, 1, 1],
                                        [2, 4, 6]], dtype=torch.float32)
        self.matrix, self.valid = positions.get_lattice_matrix(self.unit_cells)
        self.inv_matrix = torch.linalg.inv(self.matrix)
        self.recip_metric_tensor = positions.get_recip_lattice_metric_tensor(self.inv_matrix)

    def test_get_peak_positions(self):
        outputs = get_peak_positions(self.crystal_systems, self.hkl, self.intensities, self.unit_cells)
        self.assertTrue(torch.allclose(outputs[0][0][0], outputs[0][0][1:]))
        self.assertFalse(torch.allclose(outputs[0][1][0], outputs[0][1][1:]))
        self.assertEqual(outputs[1].shape, (self.unit_cells.shape[0], 3, 3))
        self.assertTrue(torch.allclose(outputs[2], self.hkl))
        self.assertTrue(torch.allclose(outputs[3], self.intensities))
        self.assertFalse(torch.allclose(outputs[5], self.unit_cells))

    def test_get_PO_intensities(self):
        peakpos = get_peak_positions(self.crystal_systems, self.hkl, self.intensities, self.unit_cells)
        rmt = peakpos[1]
        dspacing = peakpos[-2]
        outputs = get_PO_intensities(self.hkl, rmt, dspacing, self.intensities)
        self.assertEqual(outputs.shape, self.intensities.unsqueeze(2).shape)
        self.assertFalse(torch.allclose(outputs, self.intensities.unsqueeze(2)))

    def test_get_peak_shape_params(self):
        peakpos = get_peak_positions(self.crystal_systems, self.hkl, self.intensities, self.unit_cells)
        twotheta = peakpos[0]
        outputs = get_peak_shape_params(twotheta)
        self.assertEqual(outputs[0].shape, outputs[1].shape)
        self.assertEqual(outputs[0].shape, twotheta.shape)
        self.assertEqual(outputs[2].shape, (twotheta.shape[0], 1, 1))

    def test_calculate_peaks(self):
        peakpos = get_peak_positions(self.crystal_systems, self.hkl, self.intensities, self.unit_cells)
        twotheta = peakpos[0]
        hwhm_gaussian, hwhm_lorentzian, shl = get_peak_shape_params(twotheta)
        outputs = calculate_peaks(self.x, twotheta, self.intensities.unsqueeze(2),
                                hwhm_gaussian, hwhm_lorentzian, shl)
        self.assertTrue(torch.all(outputs.max(dim=-1).values <= self.intensities))
        self.assertEqual(outputs.shape[-1], self.x.shape[0])
        self.assertTrue(outputs.min() >= 0.0)

    def test_combine_peaks_for_full_patterns(self):
        peakpos = get_peak_positions(self.crystal_systems, self.hkl, self.intensities, self.unit_cells)
        twotheta = peakpos[0]
        hwhm_gaussian, hwhm_lorentzian, shl = get_peak_shape_params(twotheta)
        peaks = calculate_peaks(self.x, twotheta, self.intensities.unsqueeze(2),
                                hwhm_gaussian, hwhm_lorentzian, shl)
        outputs = combine_peaks_for_full_patterns(self.x, self.full_data, twotheta,
                                        peaks, ttmin=self.ttmin, ttmax=self.ttmax)
        self.assertEqual(outputs.shape[-1], self.datadim)
        self.assertEqual(outputs[1].min(), 0.0)
        self.assertEqual(outputs[1].max(), 1.0)
        # Expect NaN for first unit cell, as no peaks found in the data range
        # Hence the normalisation to range 0 - 1 will fail and return NaNs
        self.assertTrue(torch.all(torch.isnan(outputs[0])))

if __name__ == '__main__':
    unittest.main()