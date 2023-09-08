import torch
import positions
import intensities
import shapes
import backgroundnoise

def get_peak_positions(crystal_systems, hkl, intensities, unit_cells,
                    perturbation_stddev=0.05, zpemin=0.03, zpemax=0.03, wavelength=1.54056):
    batchsize = intensities.shape[0]
    dtype = intensities.dtype
    device = intensities.device

    cell_perturbation = positions.get_unit_cell_perturbation(crystal_systems,
                                    dtype=dtype, stddev=perturbation_stddev)
    new_unit_cells = unit_cells + cell_perturbation
    lattice_matrix, valid = positions.get_lattice_matrix(new_unit_cells)

    # Get rid of any invalid unit cells after perturbation
    if valid.sum() != valid.shape[0]:
        import warnings
        warnings.warn("Invalid cells generated")
        lattice_matrix = lattice_matrix[valid]
        hkl = hkl[valid]
        intensities = intensities[valid]
        batchsize = intensities.shape[0]
    reciprocal_lattice_matrix = torch.linalg.inv(lattice_matrix)
    reciprocal_lattice_metric_tensor = positions.get_recip_lattice_metric_tensor(reciprocal_lattice_matrix)
    d_spacing = positions.get_d_spacing(reciprocal_lattice_metric_tensor, hkl)
    zpe = positions.get_zero_point_error(batchsize, device, dtype, zpemin=zpemin, zpemax=zpemax)
    twotheta = zpe + d_to_tt(d_spacing, wavelength)

    return twotheta, reciprocal_lattice_metric_tensor, hkl, intensities, d_spacing

def get_PO_intensities(hkl, reciprocal_lattice_metric_tensor, dspacing, intensities, PO_std=0.1):
    # Now apply PO perturbation to the peak intensities
    cosP, sinP, MDfactor, PO_axis = intensities.get_MD_PO_components(hkl,
                                    reciprocal_lattice_metric_tensor, dspacing, factor_std=PO_std)
    intensities = intensities.apply_MD_PO_correction(intensities, cosP, sinP, MDfactor)
    return torch.nan_to_num(intensities)

def get_peak_shape_params(twotheta, U_min=0.0001, U_max=0.0004,
                        V_min=0.0001, V_max=0.0004, W_min=0.0001, W_max=0.0004,
                        Z_min=0.0001, Z_max=0.0004, X_min=0.001, X_max=0.035,
                        Y_min=0.001, Y_max=0.035, shlmax=0.5):
    batchsize = twotheta.shape[0]
    dtype = twotheta.dtype
    device = twotheta.device
    tan_twotheta = torch.tan(twotheta*torch.pi/180.)
    cos_twotheta = torch.cos(twotheta*torch.pi/180.)
    U, V, W, Z = shapes.get_UVWZ(batchsize, device, dtype, U_min=U_min, U_max=U_max,
                        V_min=V_min, V_max=V_max, W_min=W_min, W_max=W_max,
                        Z_min=Z_min, Z_max=Z_max)
    X, Y = shapes.get_XY(batchsize, device, dtype, X_min=X_min, X_max=X_max,
                        Y_min=Y_min,Y_max=Y_max)
    hwhm_gaussian = shapes.get_hwhm_G(tan_twotheta, cos_twotheta, U, V, W, Z)
    hwhm_lorentzian = shapes.get_hwhm_L(tan_twotheta, cos_twotheta, X, Y)
    shl = shapes.get_shl(batchsize, device, dtype, shlmax=shlmax, rescale=True)
    return hwhm_gaussian, hwhm_lorentzian, shl

def calculate_peaks(x, twotheta, intensities, hwhm_gaussian, hwhm_lorentzian, shl):
    peak_G =  shapes.gaussian(x, torch.zeros_like(twotheta), hwhm_gaussian)
    peak_L = shapes.lorentzian(x, torch.zeros_like(twotheta), hwhm_lorentzian)
    peak_FCJ = shapes.fcj(x,twotheta, shl)
    peak_GLF = torch.stack([peak_G,peak_L,peak_FCJ],dim=1)
    prod_FT_GLF = torch.fft.fft(peak_GLF).prod(dim=1)
    peak_voigt = torch.fft.ifft(prod_FT_GLF).real
    zero_sum = peak_FCJ.sum(dim=-1) == 1
    peak_voigt[zero_sum.squeeze()] = torch.fft.ifftshift(peak_voigt[zero_sum.squeeze()], dim=-1)
    peak_voigt /= peak_voigt.max(dim=2).values.unsqueeze(2)
    peak_voigt *= intensities
    return peak_voigt

def calculate_full_patterns(x, full_data, twotheta, peak_voigt, ttmin=4., ttmax=44.):
    # Finally calculate the full diffraction pattern
    twotheta[twotheta == 0] = torch.inf
    twotheta[twotheta < 4] = torch.inf
    twotheta[twotheta > 44] = torch.inf
    peakidx = torch.abs((x[0] + twotheta) - full_data).min(dim=-1).indices
    full_pattern = torch.zeros(list(peakidx.shape)+[full_data.shape[0]], device=device, dtype=dtype)
    full_pattern = full_pattern.scatter_(2,
                    peakidx.unsqueeze(2) + torch.arange(x.shape[0], device=device), peak_voigt*torch.isfinite(twotheta))

    full_pattern = full_pattern.sum(dim=1)
    full_pattern /= full_pattern.max(dim=1).values.unsqueeze(1)
    full_pattern = full_pattern[:,(full_data >= ttmin) & (full_data <= ttmax)]
    return full_pattern

def calculate_diffraction_patterns(x, full_data, crystal_systems, hkl,
                                intensities, unit_cells, wavelength=1.54056,
                                ttmin=4, ttmax=44):
    """
    Expect the input tensors to have their first dimension to be of size batchsize
    """

    twotheta, reciprocal_lattice_metric_tensor, hkl, intensities, d_spacing = get_peak_positions(crystal_systems, hkl, intensities, unit_cells,
                    perturbation_stddev=0.05, zpemin=0.03, zpemax=0.03, wavelength=wavelength)

    twotheta = twotheta.unsqueeze(2)

    mod_intensities = get_PO_intensities(hkl, reciprocal_lattice_metric_tensor, d_spacing, intensities).unsqueeze(2)

    hwhm_gaussian, hwhm_lorentzian, shl = get_peak_shape_params(twotheta)

    peak_voigt = calculate_peaks(x, twotheta, mod_intensities, hwhm_gaussian, hwhm_lorentzian, shl)

    calculated_patterns = calculate_full_patterns(x, full_data, twotheta, peak_voigt, ttmin=ttmin, ttmax=ttmax)

    bgs = backgroundnoise.get_background(calculated_patterns.shape[0],
                        full_data[(full_data >= ttmin) & (full_data <= ttmax)],
                        degree=10)
    noise = backgroundnoise.get_noise(calculated_patterns)

    calculated_patterns += bgs + noise
    calculated_patterns -= calculated_patterns.min(dim=1).values.unsqueeze(1)
    calculated_patterns /= calculated_patterns.max(dim=1).values.unsqueeze(1)

    return calculated_patterns