import unittest
import torch
from ..shapes import gaussian, lorentzian, fcj, get_UVWZ, get_XY, get_hwhm_G, get_hwhm_L, get_shl


class TestShapesFunctions(unittest.TestCase):

    def test_gaussian_valid_input(self):
        """Test the gaussian function with valid input."""
        x = torch.linspace(-10, 10, steps=101, dtype=torch.float32)
        mu = torch.tensor([[[0.], [5.]]], dtype=torch.float32)
        sig = torch.tensor([[[1.], [2.]]], dtype=torch.float32)
        peak = gaussian(x, mu, sig)
        self.assertEqual(peak.shape, (1, 2, 101))
        self.assertIsInstance(peak, torch.Tensor)

    def test_gaussian_peak_properties(self):
        """Test the properties of the gaussian peak."""
        x = torch.linspace(-1, 1, steps=101, dtype=torch.float32)
        mu = torch.tensor([[[0.]]], dtype=torch.float32)
        sig = torch.tensor([[[1.]]], dtype=torch.float32)
        peak = gaussian(x, mu, sig)
        max_index = torch.argmax(peak, dim=-1)
        self.assertTrue(torch.allclose(x[max_index].squeeze(), mu.squeeze()))

    def test_lorentzian_valid_input(self):
        """Test the lorentzian function with valid input."""
        x = torch.linspace(-10, 10, steps=101, dtype=torch.float32)
        loc = torch.tensor([[[0.], [5.]]], dtype=torch.float32)
        gam = torch.tensor([[[1.], [2.]]], dtype=torch.float32)
        peak = lorentzian(x, loc, gam)
        self.assertEqual(peak.shape, (1, 2, 101))
        self.assertIsInstance(peak, torch.Tensor)

    def test_lorentzian_peak_properties(self):
        """Test the properties of the lorentzian peak."""
        x = torch.linspace(-1, 1, steps=101, dtype=torch.float32)
        loc = torch.tensor([[[0.]]], dtype=torch.float32)
        gam = torch.tensor([[[1.]]], dtype=torch.float32)
        peak = lorentzian(x, loc, gam)
        max_index = torch.argmax(peak, dim=-1)
        self.assertTrue(torch.allclose(x[max_index].squeeze(), loc.squeeze()))

    def test_fcj_valid_input(self):
        """Test the FCJ function with valid input."""
        data = torch.linspace(1, 11, steps=101, dtype=torch.float32)
        twotheta = torch.tensor([[[2.], [3.]]], dtype=torch.float32)
        shl = torch.tensor([[0.1]], dtype=torch.float32)
        peak = fcj(data, twotheta, shl)
        self.assertEqual(peak.shape, (1, 2, 101))

    def test_fcj_zero_asymmetry(self):
        """Test the FCJ function with no asymmetry."""
        data = torch.linspace(1, 11, steps=101, dtype=torch.float32)
        twotheta = torch.tensor([[[2.], [3.]]], dtype=torch.float32)
        shl = torch.tensor([[0.]], dtype=torch.float32)
        peak = fcj(data, twotheta, shl)
        self.assertEqual(peak.sum(), 2.)
        self.assertEqual(peak[:,:,0].sum(), 2.)

    def test_get_UVWZ_valid_input(self):
        """Test get_UVWZ with valid input."""
        batchsize = 10
        dtype = torch.float32
        device = torch.device("cpu")
        U_min=0.0001
        U_max=0.0004
        V_min=0.0001
        V_max=0.0004
        W_min=0.0001
        W_max=0.0004
        Z_min=0.0001
        Z_max=0.0004
        UVWZ = get_UVWZ(batchsize, device, dtype, U_min=U_min, U_max=U_max,
                        V_min=V_min, V_max=V_max, W_min=W_min, W_max=W_max,
                        Z_min=Z_min, Z_max=Z_max)
        self.assertEqual(len(UVWZ), 4)
        self.assertEqual(UVWZ[0].shape[0], batchsize)
        self.assertEqual(UVWZ[0].device, device)
        self.assertEqual(UVWZ[1].dtype, dtype)
        self.assertTrue((UVWZ[0].min() >= U_min) and (UVWZ[0].max() <= U_max))
        self.assertTrue((UVWZ[1].min() >= V_min) and (UVWZ[1].max() <= V_max))
        self.assertTrue((UVWZ[2].min() >= W_min) and (UVWZ[2].max() <= W_max))
        self.assertTrue((UVWZ[3].min() >= Z_min) and (UVWZ[3].max() <= Z_max))

    def test_get_XY_valid_input(self):
        """Test get_XY with valid input."""
        batchsize = 10
        dtype = torch.float32
        device = torch.device("cpu")
        X_min=0.001
        X_max=0.035
        Y_min=0.001
        Y_max=0.035
        XY = get_XY(batchsize, device, dtype, X_min=X_min, X_max=X_max,
                    Y_min=Y_min, Y_max=Y_max)
        self.assertEqual(len(XY), 2)
        self.assertEqual(XY[0].shape[0], batchsize)
        self.assertEqual(XY[0].device, device)
        self.assertEqual(XY[1].dtype, dtype)
        self.assertTrue((XY[0].min() >= X_min) and (XY[0].max() <= X_max))
        self.assertTrue((XY[1].min() >= Y_min) and (XY[1].max() <= Y_max))

    def test_get_hwhm_G(self):
        """Test get_hwhm_G with valid input."""
        batchsize = 2
        device = torch.device("cpu")
        dtype = torch.float32
        U, V, W, Z = get_UVWZ(batchsize, device, dtype)
        twotheta = torch.tensor([[[5.], [10.], [22.]], [[7.], [21.], [40.]]])
        tan_twotheta = torch.tan(torch.deg2rad(twotheta))
        cos_twotheta = torch.cos(torch.deg2rad(twotheta))
        hwhm_G = get_hwhm_G(tan_twotheta, cos_twotheta, U, V, W, Z)
        self.assertEqual(hwhm_G.shape[0], batchsize)
        self.assertEqual(hwhm_G.shape, (batchsize, 3, 1))

    def test_get_hwhm_L(self):
        """Test get_hwhm_L with valid input."""
        batchsize = 2
        device = torch.device("cpu")
        dtype = torch.float32
        X, Y = get_XY(batchsize, device, dtype)
        twotheta = torch.tensor([[[5.], [10.], [22.]], [[7.], [21.], [40.]]])
        tan_twotheta = torch.tan(torch.deg2rad(twotheta))
        cos_twotheta = torch.cos(torch.deg2rad(twotheta))
        hwhm_L = get_hwhm_L(tan_twotheta, cos_twotheta, X, Y)
        self.assertEqual(hwhm_L.shape[0], batchsize)
        self.assertEqual(hwhm_L.shape, (batchsize, 3, 1))

    def test_get_shl(self):
        """Test get_shl with valid input."""
        batchsize = 2
        device = torch.device("cpu")
        dtype = torch.float32
        shlmax = 0.5
        torch.random.manual_seed(0)
        shl = get_shl(batchsize, device, dtype, shlmax=shlmax)
        torch.random.manual_seed(0)
        shl_no_rescale = get_shl(batchsize, device, dtype, shlmax=shlmax, rescale=False)
        self.assertEqual(shl.shape[0], batchsize)
        self.assertEqual(shl_no_rescale.shape[0], batchsize)
        self.assertTrue(torch.allclose(shl, shl_no_rescale/57.2958))

if __name__ == '__main__':
    unittest.main()
