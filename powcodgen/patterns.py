import torch
import time
import warnings
from powcodgen import positions
from powcodgen import intensity
from powcodgen import shapes
from powcodgen import backgroundnoise


def get_initial_tensors(ttmin=4., ttmax=44., peakrange=3, datadim=2048,
                        device=torch.device("cpu"), dtype=torch.float32):
    """Generates the baseline tensors for calculating the diffraction patterns

    Args:
        ttmin (float, optional): Minimum angle for the PXRD data. Defaults to 4.
        ttmax (float, optional): Maximum angle for the PXRD data. Defaults to 44.
        peakrange (int, optional): The distance (in degrees) around the centre
                                    of each peak to generate the peak within.
                                    i.e. if the peak is positioned at 15 degrees
                                    there is little point calculating the
                                    contribution to the intensity at 2 degrees.
                                    Instead, calculate the intensity contribution
                                    +/- peakrange around the centre of the peak.
                                    Defaults to 3.
        datadim (int, optional): The dimensionality of data, i.e. the number of
                                points in the PXRD histograms. Defaults to 2048.
        device (torch.device, optional): The device to run the calculations on.
                                        Defaults to torch.device("cpu").
        dtype (torch.dtype, optional): The datatype for the tensors.
                                        Defaults to torch.float32.

    Returns:
        Tuple: pytorch tensors for the full data range with an extra
                +/- 0.5*peakrange at either end, the peakrange and an
                array for the diffraction data suitable for plotting with
                matplotlib
    """
    full_data = torch.linspace(ttmin-(peakrange/2), ttmax+(peakrange/2),
        int(torch.ceil(torch.tensor((ttmax-ttmin+peakrange)/((ttmax-ttmin)/datadim)))),
        device=device, dtype=dtype)
    plotdata = full_data[(full_data >= ttmin) & (full_data <= ttmax)].cpu()
    x = (full_data[full_data <= ttmin+(peakrange/2)]).clone() - ttmin
    return full_data, x, plotdata

def get_peak_positions(crystal_systems, hkl, intensities, unit_cells,
                    perturbation_stddev=0.05, zpemin=-0.03, zpemax=0.03,
                    wavelength=1.54056):
    """Generate peak positions on the basis of unit cells and Miller indices.
    This will automatically perturb the unit cells, keeping their symmetries,
    and will remove any invalid unit cells. Hence, this function also needs the
    intensities so that the intensities for invalid cells can be ignored.

    Args:
        crystal_systems (tensor): The crystal systems for the unit cells. These
            are numbers from 0 - 7, which encode the crystal system in
            alphabetical order, i.e. cubic = 0, trigonal_rhombohedral = 7.
            Shape = (batch, 1)
        hkl (tensor): Miller indices for the reflections.
            Shape = (batch, n_peaks, 3)
        intensities (tensor): The intensities for the reflections.
            Shape = (batch, n_peaks)
        unit_cells (tensor): The unit cells for the crystals, expressed as
            lengths and angles in degrees, i.e. [a, b, c, al, be, ga].
            Shape = (batch, 6)
        perturbation_stddev (float, optional): The standard deviation to use for
            the unit cell perturbations. Defaults to 0.05.
        zpemin (float, optional): The minimum value for the zero-point error.
            Defaults to -0.03.
        zpemax (float, optional): The maximum value for the zero-point error.
            Defaults to 0.03.
        wavelength (float, optional): The wavelength for the data generation.
            Defaults to 1.54056 (Cu Ka1)

    Returns:
        Tuple: A tuple of tensors containing the two theta positions
            (batch, npeaks, 1), the reciprocal lattice metric tensors for the
            perturbed unit cells (batch, 3, 3), the Miller indices for the
            reflections (batch, n_peaks, 3), the d-spacings for the reflections
            (batch, n_peaks, 1), the perturbed unit cells (batch, 6)
    """
    batchsize = intensities.shape[0]
    dtype = intensities.dtype
    device = intensities.device

    cell_perturbation = positions.get_unit_cell_perturbation(crystal_systems,
                                    dtype=dtype, stddev=perturbation_stddev)
    new_unit_cells = unit_cells + cell_perturbation
    lattice_matrix, valid = positions.get_lattice_matrix(new_unit_cells)

    # Get rid of any invalid unit cells after perturbation and warn the user
    # that this has happened.
    if valid.sum() != valid.shape[0]:
        warnings.warn("Invalid cells generated")
        lattice_matrix = lattice_matrix[valid]
        hkl = hkl[valid]
        intensities = intensities[valid]
        batchsize = intensities.shape[0]
    reciprocal_lattice_matrix = torch.linalg.inv(lattice_matrix)
    reciprocal_lattice_metric_tensor = positions.get_recip_lattice_metric_tensor(
                                                    reciprocal_lattice_matrix)
    d_spacing = positions.get_d_spacing(reciprocal_lattice_metric_tensor, hkl)
    zpe = positions.get_zero_point_error(batchsize, device, dtype, zpemin=zpemin,
                                        zpemax=zpemax)
    twotheta = zpe + positions.d_to_tt(d_spacing, wavelength)
    twotheta = twotheta.unsqueeze(2)
    return twotheta, reciprocal_lattice_metric_tensor, hkl, intensities, d_spacing, new_unit_cells

def get_PO_intensities(hkl, reciprocal_lattice_metric_tensor, dspacing, intensities,
                        PO_std=0.1):
    """Modify a set of input intensities with the March Dollase preferred orientation
    correction. This function will generate the MD parameters so the user does not
    need to do this manually. If manual control of the MD parameters is desired,
    then generate the cosP, sinP and MD factors, then use the apply_MD_PO_correction
    function from the intensities module.

    Args:
        hkl (tensor): The Miller indices for the reflections. Shape = (batch, n_peaks, 3)
        reciprocal_lattice_metric_tensor (tensor): The reciprocal lattice metric tensors
            for the unit cells. Shape = (batch, 3, 3)
        dspacing (tensor): The d-spacings from the reflections. Shape = (batch, n_peaks)
        intensities (tensor): The intensities for the reflections. Shape = (batch, n_peaks)
        PO_std (float, optional): The standard deviation to use to generate the
            March Dollase factors. Defaults to 0.1.

    Returns:
        tensor: the PO-modified intensities. Shape = (batch, n_peaks, 1)
    """
    cosP, sinP, MDfactor, PO_axis = intensity.get_MD_PO_components(hkl,
                                    reciprocal_lattice_metric_tensor, dspacing,
                                    factor_std=PO_std)
    intensities = intensity.apply_MD_PO_correction(intensities, cosP, sinP,
                                                    MDfactor)
    return torch.nan_to_num(intensities)

def get_peak_shape_params(twotheta, U_min=0.0001, U_max=0.0005,
                        V_min=0.0001, V_max=0.0005, W_min=0.0001, W_max=0.0005,
                        Z_min=0.0001, Z_max=0.0005, X_min=0.001, X_max=0.04,
                        Y_min=0.001, Y_max=0.04, shlmax=0.5):
    """Generate the HWHMs for the gaussian and lorentzian contributions to the peaks
    as well as generating the asymmetry parameters used in the Finger, Cox and
    Jephcoat axial divergence profile function.

    Args:
        twotheta (tensor): The two-theta positons for the peaks.
            Shape = (batch, n_peaks, 1)
        U, V, W, X, Y, Z are the peak shape parameters described in this article:
            https://journals.iucr.org/j/issues/2021/06/00/gj5272/
            Briefly, U, V, W and Z contribute to the gaussian HWHM, whilst X and
            Y contribute to the lortentzian HWHM. U and X related to microstrain
            and Z and Y are related to domain size. The default parameters
            chosen help to generate peaks with FWHMs close to those seen with
            laboratory diffraction data.
        shlmax (float, optional): The asymmetry parameter for the FCJ axial
            divergence model. This follows the naming and implementation used by
            GSAS-II. For more detail, see here:
            https://gsas-ii.readthedocs.io/en/latest/_modules/GSASIIpwd.html#fcjde_gen
            Defaults to 0.5.

    Returns:
        tuple: tuple of tensors containing the HWHM for the gaussian and
            lorentzian components, as well as the asymnmetry parameters for each
            of the peaks. The HWHM tensors each have shape (batch, n_peaks, 1).
            The SHL tensor has shape (batch, 1, 1).
    """
    batchsize = twotheta.shape[0]
    dtype = twotheta.dtype
    device = twotheta.device
    tan_twotheta = torch.tan(twotheta*torch.pi/180.)
    cos_twotheta = torch.cos(twotheta*torch.pi/180.)
    U, V, W, Z = shapes.get_UVWZ(batchsize, device, dtype, U_min=U_min,
                        U_max=U_max, V_min=V_min, V_max=V_max, W_min=W_min,
                        W_max=W_max, Z_min=Z_min, Z_max=Z_max)
    X, Y = shapes.get_XY(batchsize, device, dtype, X_min=X_min, X_max=X_max,
                        Y_min=Y_min,Y_max=Y_max)
    hwhm_gaussian = shapes.get_hwhm_G(tan_twotheta, cos_twotheta, U, V, W, Z)
    hwhm_lorentzian = shapes.get_hwhm_L(tan_twotheta, cos_twotheta, X, Y)
    shl = shapes.get_shl(batchsize, device, dtype, shlmax=shlmax, rescale=True)
    return hwhm_gaussian, hwhm_lorentzian, shl

def calculate_peaks(x, twotheta, intensities, hwhm_gaussian, hwhm_lorentzian, shl):
    """Calculates the diffraction peaks by convolving gaussian, lorentzian and
    FCJ profile functions to produce asymmetric Voigt profiles.
    Convolution is done by applying a FFT to the individual peaks, then
    doing pointwise multiplication of the FFT transformed peaks, then taking
    the inverse FFT.

    Args:
        x (tensor): the data around which the peak will be generated. This is
            produced in the get_initial_tensors function, and will by default be
            centred at zero, with +/- 1.5 degrees twotheta either side.
            Shape = (n_points_for_peak)
        twotheta (tensor): The positions of the reflections.
            Shape = (batch, n_peaks, 1)
        intensities (tensor): The intensities of the reflections.
            Shape = (batch, n_peaks, 1)
        hwhm_gaussian (tensor): The HWHMs for the gaussian components.
            Shape = (batch, n_peaks, 1)
        hwhm_lorentzian (tensor): The HWHMs for the lorentzian components.
            Shape = (batch, n_peaks, 1)
        shl (tensor): The asymmetry parameter for the patterns.
            Shape = (batch, 1, 1)

    Returns:
        tensor: the Voigt peaks. Shape = (batch, n_peaks, n_points_for_peak)
    """
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

def combine_peaks_for_full_patterns(x, full_data, twotheta, peak_voigt, ttmin=4., ttmax=44.):
    """Combine several Voigt peaks into a full diffraction pattern.

    Args:
        x (tensor): x (tensor): the data around which the peak will be generated.
            This is produced in the get_initial_tensors function, and will by
            default be centred at zero, with +/- 1.5 degrees twotheta either side.
            Shape = (n_points_for_peak)
        full_data (tensor): The full datarange, + an extra bit either side, to
            allow for peaks that are generated close to the limit of the data
            range to impact the intensity within the data range.
            Shape = (datadim + n_points_for_peak)
        twotheta (tensor): The twotheta positions of the peaks.
            Shape = (batch, n_peaks, 1)
        peak_voigt (tensor): The Voigt profiles for each of the peaks.
            Shape = (batch, n_peaks, n_points_for_peak)
        ttmin (float, optional): Minimum twotheta angle for the data.
            Defaults to 4.
        ttmax (float, optional): Maximum twotheta angle for the data.
            Defaults to 44.

    Returns:
        tensor: The full diffraction patterns, normalised such that the minimum
            intensity is zero, and the maximum is 1 for each of the patterns.
            Shape = (batch, datadim)
    """
    twotheta[twotheta == 0] = torch.inf
    twotheta[twotheta < ttmin] = torch.inf
    twotheta[twotheta > ttmax] = torch.inf
    device = x.device
    dtype = x.dtype
    # Find the start index for where the Voigt profiles should begin in the full
    # data range.
    peakidx = torch.abs((x[0] + twotheta) - full_data).min(dim=-1).indices
    # Initialize the full patterns with zero intensity, then add the Voigt profiles.
    # It will initially have shape (batch, n_peaks, datadim + n_points_for_peak).
    full_pattern = torch.zeros(list(peakidx.shape)+[full_data.shape[0]],
                            device=device, dtype=dtype)
    # The scatter_ function takes the individual profiles from the peak_voigt tensor
    # and writes them into the appropriate place in the full_pattern tensor. The
    # result is then summed down the n_peaks axis (1) to give the full diffraction
    # patterns, with shape (batch, datadim + n_points_for_peak)
    full_pattern = full_pattern.scatter_(2,
                    peakidx.unsqueeze(2) + torch.arange(x.shape[0], device=device),
                    peak_voigt*torch.isfinite(twotheta))
    full_pattern = full_pattern.sum(dim=1)
    # Normalize the full_patterns, then cut the data into the twotheta range desired.
    full_pattern -= full_pattern.min(dim=1).values.unsqueeze(1)
    full_pattern /= full_pattern.max(dim=1).values.unsqueeze(1)
    full_pattern = full_pattern[:,(full_data >= ttmin) & (full_data <= ttmax)]
    return full_pattern

def calculate_diffraction_patterns(x, full_data, crystal_systems, hkl,
                                intensities, unit_cells, wavelength=1.54056,
                                ttmin=4., ttmax=44.):
    """
    Expect the input tensors to have their first dimension to be of size batchsize
    """

    twotheta, reciprocal_lattice_metric_tensor, hkl, intensities, d_spacing, new_unit_cells = get_peak_positions(
        crystal_systems, hkl, intensities, unit_cells, perturbation_stddev=0.05,
        zpemin=-0.03, zpemax=0.03, wavelength=wavelength)

    mod_intensities = get_PO_intensities(hkl, reciprocal_lattice_metric_tensor,
                                        d_spacing, intensities)

    hwhm_gaussian, hwhm_lorentzian, shl = get_peak_shape_params(twotheta)

    peak_voigt = calculate_peaks(x, twotheta, mod_intensities, hwhm_gaussian,
                                hwhm_lorentzian, shl)

    calculated_patterns = combine_peaks_for_full_patterns(x, full_data, twotheta,
                                            peak_voigt, ttmin=ttmin, ttmax=ttmax)

    bgs = backgroundnoise.get_background(calculated_patterns.shape[0],
                        full_data[(full_data >= ttmin) & (full_data <= ttmax)],
                        degree=10)
    noise = backgroundnoise.get_noise(calculated_patterns)

    # Final scaling to range 0-1
    calculated_patterns_bg_noise = calculated_patterns + bgs + noise
    calculated_patterns_bg_noise -= calculated_patterns_bg_noise.min(dim=1).values.unsqueeze(1)
    calculated_patterns_bg_noise /= calculated_patterns_bg_noise.max(dim=1).values.unsqueeze(1)

    notnan = (torch.isnan(calculated_patterns_bg_noise).sum(dim=-1) == 0)


    return calculated_patterns_bg_noise[notnan], calculated_patterns[notnan]

def calculate_diffraction_patterns_with_impurities(x, full_data, crystal_systems,
    hkl, intensities, unit_cells, wavelength=1.54056, ttmin=4, ttmax=44,
    same_hwhm=True, max_impurity_intensity=0.15, min_impurity_intensity=0.01,
    add_background=True, shuffle_seed=None, add_noise=True, start_mask=True):
    """
    Expect the input tensors to have their first dimension to be of size batchsize
    The first third of the batch will be used for the pure patterns
    The second third of the batch will be used as the dominant phases
    The final third of the batch will be used as minority (impurity) phases

    The resultant combined data will then be shuffled to ensure that the network
    doesn't learn this pattern. For consistent shuffling, you can supply the
    "shuffle_seed" parameter.
    """
    batchsize = intensities.shape[0]
    one_third = int(batchsize // 3)
    device = x.device
    dtype = x.dtype
    indices = torch.arange(batchsize, dtype=torch.long, device=device)
    pure = indices[:one_third]
    dominant = indices[one_third:2*one_third]
    minority = indices[2*one_third:]
    twotheta, reciprocal_lattice_metric_tensor, hkl, intensities, d_spacing, new_unit_cells = get_peak_positions(
        crystal_systems, hkl, intensities, unit_cells, perturbation_stddev=0.05,
        zpemin=-0.03, zpemax=0.03, wavelength=wavelength)

    mod_intensities = get_PO_intensities(hkl, reciprocal_lattice_metric_tensor,
                                        d_spacing, intensities).unsqueeze(2)

    hwhm_gaussian, hwhm_lorentzian, shl = get_peak_shape_params(twotheta)

    # if peaks are set to have the same hwhm, need to modify the peak_shape_params
    # to ensure that the impurity lines have the same hwhm. Regardless of this
    # setting, the asymmetry (shl) parameter must be the same for a given pattern
    # as this is of instrumental rather than sample based origin
    if same_hwhm:
        hwhm_gaussian[minority] = hwhm_gaussian[dominant]
        hwhm_lorentzian[minority] = hwhm_lorentzian[dominant]
    shl[minority] = shl[dominant]

    peak_voigt = calculate_peaks(x, twotheta, mod_intensities, hwhm_gaussian,
                                hwhm_lorentzian, shl)

    calculated_patterns = combine_peaks_for_full_patterns(x, full_data, twotheta,
                                        peak_voigt, ttmin=ttmin, ttmax=ttmax)

    # At this point, we want to scale down the minority phase intensities before
    # we add the minority phase pattern to the dominant phase pattern
    impure_intensities = torch.rand((one_third, 1),
                            device=intensities.device, dtype=intensities.dtype)
    impure_intensities *= (max_impurity_intensity - min_impurity_intensity)
    impure_intensities += min_impurity_intensity

    pure_data = calculated_patterns[pure]
    minority_data = impure_intensities * calculated_patterns[minority]
    dominant_data = calculated_patterns[dominant]
    impure_data = dominant_data + minority_data
    impure_data -= impure_data.min(dim=1).values.unsqueeze(1)
    impure_data /= impure_data.max(dim=1).values.unsqueeze(1)
    zero_impurity = torch.zeros_like(pure_data)

    combined_patterns = torch.cat([pure_data, impure_data], dim=0)
    first_nonzero = torch.argmax((combined_patterns > 0).type(torch.int),dim=-1)
    pure_patterns = torch.cat([pure_data, dominant_data], dim=0)
    impure_patterns = torch.cat([zero_impurity, minority_data], dim=0)

    bgs = backgroundnoise.get_background(combined_patterns.shape[0],
                        full_data[(full_data >= ttmin) & (full_data <= ttmax)],
                        degree=10)
    noise = backgroundnoise.get_noise(combined_patterns)

    # Final scaling to range 0-1
    if add_background:
        combined_patterns += bgs
    if add_noise:
        combined_patterns += noise
    combined_patterns -= combined_patterns.min(dim=1).values.unsqueeze(1)
    combined_patterns /= combined_patterns.max(dim=1).values.unsqueeze(1)

    if shuffle_seed is not None:
        torch.manual_seed(shuffle_seed)
    shuffle = torch.randperm(combined_patterns.shape[0])
    if shuffle_seed is not None:
        # Reset the random seed using the time for further work with pytorch
        torch.manual_seed(int(time.time()*10000000))
    impure = torch.ones(one_third, device=device, dtype=dtype)
    pure = torch.zeros_like(impure)
    pure_impure = torch.cat([pure, impure], dim=0)

    combined_patterns = combined_patterns[shuffle]
    first_nonzero = first_nonzero[shuffle]
    pure_patterns = pure_patterns[shuffle]
    impure_patterns = impure_patterns[shuffle]
    pure_impure = pure_impure[shuffle]
    cs = crystal_systems[:2*one_third][shuffle]
    cell = new_unit_cells[:2*one_third][shuffle]

    notnan = (torch.isnan(combined_patterns).sum(dim=-1) == 0)
    combined_patterns = combined_patterns[notnan]
    first_nonzero = first_nonzero[notnan]
    impure_patterns = impure_patterns[notnan]
    pure_patterns = pure_patterns[notnan]
    pure_impure = pure_impure[notnan]
    cs = cs[notnan]
    cell = cell[notnan]
    if torch.isnan(combined_patterns).sum() > 1:
        print("NaNs in training data generation - ",
                torch.isnan(combined_patterns).sum(dim=-1))

    # Zero out the start of the pattern, and set those elements equal to the value
    # of the first non-zero element, as an additional source of augmentation.
    # This will only happen in half of the batches.
    if (torch.randint(0,2,(1,)).sum() > 0) and start_mask:
        neg_mask = (torch.arange(combined_patterns.shape[1],
                    device=device).unsqueeze(1) < first_nonzero.squeeze()).type(dtype).T
        pos_mask = 1-neg_mask
        combined_patterns *= pos_mask
        combined_patterns += neg_mask * torch.gather(combined_patterns,1,first_nonzero.unsqueeze(1))

    return combined_patterns, pure_patterns, impure_patterns, pure_impure, cs, cell

