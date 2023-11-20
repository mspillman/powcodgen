import torch


def gaussian(x, mu, sig):
    """Calculate a gaussian peak

    Args:
        x (tensor): the x-coordinates for the peak. Shape = (datapoints)
        mu (tensor): the mean of the gaussian.
            Shape = (batch, number of peaks, 1)
        sig (tensor): the standard deviation of the gaussian.
            Shape = (batch, number of peaks, 1)

    Returns:
        tensor: the gaussian peaks centered at mu with standard deviation sig.
            Intensities scaled to max = 1.
            Shape = (batch, number of peaks, datapoints)
    """
    root_two_pi = 2.5066282746310002
    peak = (1/(sig*root_two_pi))*torch.exp(-0.5*(x-mu)**2/sig**2)
    return peak/peak.max(dim=-1).values.unsqueeze(2)


def lorentzian(x, loc, gam):
    """Calculate a lorentzian peak

    Args:
        x (tensor): the x-coordinates for the peak. Shape = (datapoints)
        loc (tensor): The centre of the peaks.
            Shape = (batch, number of peaks, 1)
        gam (tensor): The half width at half max for the peaks.
             Shape = (batch, number of peaks, 1)

    Returns:
        tensor: the gaussian peaks centered at loc with HWHM gam.
            Intensities scaled to max = 1.
            Shape = (batch, number of peaks, datapoints)
    """
    one_over_pi = 0.3183098861837907
    peak = one_over_pi*(gam/((x - loc)**2+gam**2))
    return peak/peak.max(dim=-1).values.unsqueeze(2)


def fcj(data,twotheta,shl):
    """Finger, Cox and Jephcoat profile function.
    Code based on GSAS-II function found here:
    https://gsas-ii.readthedocs.io/en/latest/_modules/GSASIIpwd.html#fcjde_gen

    Args:
        data (tensor): the data to evaluate the function on.
            Shape = (batch, number of peaks, datapoints)
        twotheta (tensor): The position of the peak.
            Shape = (batch, number of peaks, 1)
        shl (tensor): shl = sum(S/L,H/L) where:
            S: sample height
            H: detector opening
            L: sample to detector opening distance
            Shape = (batch, 1)

    Returns:
        tensor: The Finger, Cox and Jephcoat profiles for the peaks
            Shape = (batch, number of peaks, datapoints)
    """
    pi_over_180 = 0.017453292519943295
    step = data[1] - data[0]
    T = step*data+twotheta
    abs_cos_T = torch.abs(torch.cos(T*pi_over_180))
    abs_cos_T_sqd = abs_cos_T**2
    cos_sqd_twotheta = torch.cos(twotheta*pi_over_180)**2
    cos_sqd_twotheta = torch.where(abs_cos_T_sqd>cos_sqd_twotheta,
                                    cos_sqd_twotheta,abs_cos_T_sqd)
    fcj_profile = torch.where(abs_cos_T_sqd>cos_sqd_twotheta,
                    (torch.sqrt(cos_sqd_twotheta/(abs_cos_T_sqd-cos_sqd_twotheta+1e-9))
                    -1./shl)/abs_cos_T,0.0)
    fcj_profile = torch.where(fcj_profile > 0.,fcj_profile,0.)
    # Sometimes the FCJ profile returned is all zeros. We don't want to separate
    # these out with loops etc, so need to deal with them. We'll replace zero entry
    # with an array like this ([1, 0, 0, ..., 0]) which gives ([1, 1, 1, ..., 1])
    # after the Fourier transform. This then means that the pointwise multiplication
    # of the FT'd Gaussian and Lorentzian components can still occur unaffected
    # This then allows the full Voigt profiles to be calculated in the absence of
    # any asymmetry.
    zero_sum = (fcj_profile.sum(dim=-1) == 0).type(fcj_profile.dtype)
    fcj_profile[:,:,0] += zero_sum
    #first_zero = torch.zeros_like(x)
    #first_zero[0] = 1
    #batch_first_zero = (zero_sum * first_zero.unsqueeze(0))
    #peak_FCJ += batch_first_zero
    return fcj_profile

def get_UVWZ(batchsize, device, dtype, U_min=0.0001, U_max=0.0004,
            V_min=0.0001, V_max=0.0004, W_min=0.0001, W_max=0.0004,
            Z_min=0.0001, Z_max=0.0004):
    """
    Get parameters for Gaussian HWHM. Defaults should give reasonable data to
    resemble laboratory diffraction data
    """
    U = ((torch.rand(batchsize, device=device, dtype=dtype) * (U_max-U_min))
            + U_min).unsqueeze(1)

    V = ((torch.rand(batchsize, device=device, dtype=dtype) * (V_max-V_min))
            + V_min).unsqueeze(1)

    W = ((torch.rand(batchsize, device=device, dtype=dtype) * (W_max-W_min))
            + W_min).unsqueeze(1)

    Z = ((torch.rand(batchsize, device=device, dtype=dtype) * (Z_max-Z_min))
            + Z_min).unsqueeze(1)

    return U, V, W, Z

def get_XY(batchsize, device, dtype, X_min=0.001, X_max=0.035,
            Y_min=0.001,Y_max=0.035):
    """
    Get parameters for Lorentzian HWHM. Defaults should give reasonable data to
    resemble laboratory diffraction data
    """
    X = ((torch.rand(batchsize, device=device, dtype=dtype) * (X_max-X_min))
            + X_min).unsqueeze(1)

    Y = ((torch.rand(batchsize, device=device, dtype=dtype) * (Y_max-Y_min))
            + Y_min).unsqueeze(1)

    return X, Y

def get_hwhm_G(tan_twotheta, cos_twotheta, U, V, W, Z):
    """Calculate Gaussian HWHM as a function of peak position and U,V,W and Z params

    Args:
        tan_twotheta (tensor): tangent of the twotheta peak positions
        cos_twotheta (tensor): cosine of the twotheta peak positions
        U (tensor): peakshape parameter U
        V (tensor): peakshape parameter V
        W (tensor): peakshape parameter W
        Z (tensor): peakshape parameter Z

    Returns:
        tensor: HWHM for the gaussian peaks
            Shape = (batch, number of peaks)
    """
    tan_twotheta = tan_twotheta.squeeze()
    cos_twotheta = cos_twotheta.squeeze()
    return torch.sqrt((U * tan_twotheta**2) + (V * tan_twotheta)
                        + W + (Z/(cos_twotheta**2))).unsqueeze(2)

def get_hwhm_L(tan_twotheta, cos_twotheta, X, Y):
    """Calculate Lorentzian HWHM as a function of peak position and X and Y params

    Args:
        tan_twotheta (tensor): tangent of the twotheta peak positions
        cos_twotheta (tensor): cosine of the twotheta peak positions
        X (tensor): peakshape parameter X
        Y (tensor): peakshape parameter Y

    Returns:
        tensor: HWHM for the lorentzian peaks
            Shape = (batch, number of peaks)
    """
    tan_twotheta = tan_twotheta.squeeze()
    cos_twotheta = cos_twotheta.squeeze()
    return ((X * tan_twotheta) + (Y/cos_twotheta)).unsqueeze(2)

def get_shl(batchsize, device, dtype, shlmax=0.5, rescale=True):
    """Generate asymmetry parameter for the FCJ profile. Adapted from GSAS-II,
    See fcjde_gen and getFCJVoigt in original GSAS-II code here:
    https://gsas-ii.readthedocs.io/en/latest/_modules/GSASIIpwd.html


    Args:
        shl = sum(S/L,H/L) where:
            S: sample height
            H: detector opening
            L: sample to detector opening distance
        This is scaled by 1/57.2958 if rescale is True, in keeping with the scaling
        applied in the original GSAS-II code

    Returns:
        tensor: The asymmetry parameter for the FCJ profiles
    """
    shl = torch.rand((batchsize, 1, 1), device=device, dtype=dtype) * shlmax
    if rescale:
        shl /= 57.2958
    return shl