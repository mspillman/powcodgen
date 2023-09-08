def get_noise(calculated_patterns, noise_min = 0.0001, noise_max = 0.0025):
    """Get noise for the diffraction patterns to simulate experimental data

    Args:
        calculated_patterns (tensor): The diffraction patterns.
            Shape = (batch, points in profile)
        noise_min (float, optional): Minimum standard deviation for the
            Gaussian noise. Defaults to 0.0001.
        noise_max (float, optional): Maximum standard deviation for the
            Gaussian noise. Defaults to 0.0025.

    Returns:
        tensor: Noise to be added to the diffraction patterns
    """
    batchsize = calculated_patterns.shape[0]
    device = calculated_patterns.device
    dtype = calculated_patterns.dtype
    noise_std = torch.rand((batchsize,1), device=device, dtype=dtype) * (noise_max - noise_min) + noise_min
    noise = torch.randn(calculated_patterns.shape, device=device, dtype=dtype) * noise_std
    return noise

def get_background(batchsize, data, bg_prm_max=0.025, bg_prm_min=0.0, degree=8):
    """Calculate background profiles using Chebyshev polynomials

    Args:
        batchsize (int): The batch size
        data (tensor): The twotheta values for the diffraction histograms
        bg_prm_max (float, optional): Maximum value for the standard deviation
            of the weights for the Chebyshev polynomial components. Defaults to
            0.025.
        bg_prm_min (float, optional): Minimum value for the standard deviation
            of the weights for the Chebyshev polynomial components. Defaults to
            0.0.
        degree (int, optional): The degree of Chebyshev polynomial to use for
            the backgrounds. Defaults to 8.

    Returns:
        tensor: Background profiles. Shape = (batch, number of points in histogram)
    """
    device = data.device
    dtype = data.dtype
    n = torch.arange(degree,device=device,dtype=dtype).unsqueeze(1)
    # Scale data into range -1 to +1
    ttstar = 2*(data - data.min())/(data.max() - data.min()) - 1
    chebyshev_basis = torch.cos(n*torch.arccos(ttstar))
    params = (((torch.rand((batchsize,1,1), device=device, dtype=dtype)
                * (bg_prm_max - bg_prm_min)) + bg_prm_min)
                * torch.randn((batchsize, chebyshev_basis.shape[0], 1),
                device=device, dtype=dtype))
    bg = (params * chebyshev_basis).sum(dim=1)
    return bg