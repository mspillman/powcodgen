import torch


def get_unit_cell_perturbation(crystal_systems, dtype=torch.float32, stddev=0.05):
    """Generate a perturbation for the unit cell lengths and angles for a given
    crystal system.

    Args:
        crystal_systems (tensor):   Crystal systems for the unit cells.
                                    Shape = (batch)
        dtype (torch.dtype, optional): Defaults to torch.float32.
        stddev (float, optional):   Standard deviation for the gaussian noise.
                                    Defaults to 0.05.

    Returns:
        tensor: A tensors to additively adjust the unit cell lengths and angles
                Shape = (batch, 6)
    """
    batchsize = crystal_systems.shape[0]
    device = crystal_systems.device
    lengths, angles = torch.randn((2, batchsize, 3), device=device, dtype=dtype) * stddev
    cubic = crystal_systems == 0
    hexagonal = crystal_systems == 1
    monoclinic = crystal_systems == 2
    #orthorhombic = crystal_systems == 3 # Don't need to query data for this
    tetragonal = crystal_systems == 4
    triclinic = crystal_systems == 5
    trigonal_h = crystal_systems == 6
    trigonal_r = crystal_systems == 7

    # Cubic, tetragonal, rhombohedral and hexagonal - a and b must be the same
    lengths[:,1] = torch.where(cubic | hexagonal | tetragonal | trigonal_h | trigonal_r,
                            lengths[:,0], lengths[:,1])
    # Cubic and rhombohedral cells - a, b and c must be the same
    lengths[:,2] = torch.where(cubic | trigonal_r, lengths[:,0], lengths[:,2])
    # Rhombohedral and triclinic cells - could change their alpha values
    angles[:,0] = torch.where((trigonal_r | triclinic), angles[:,0], 0.)
    # Triclinic or monoclinic cells could change beta values
    angles[:,1] = torch.where((triclinic | monoclinic), angles[:,1], 0.)
    # Triclinc cells could change gamma
    angles[:,2] = torch.where(triclinic, angles[:,2], 0.)
    # Rhombohedral cells - need to ensure all angles are the same
    angles[:,1] = torch.where(trigonal_r, angles[:,0], angles[:,1])
    angles[:,2] = torch.where(trigonal_r, angles[:,0], angles[:,2])

    return torch.concat([lengths, angles], dim=-1)


def get_lattice_matrix(unit_cell_dimensions):
    """calculate a lattice matrix from unit cell dimensions

    Args:
        unit_cell_dimensions (tensor):  The unit cell dimensions. Lengths in
                                        angstroms, angles in degrees.
                                        Shape = (batch, 6)

    Returns:
        tensor: The matrix representation of the unit cells.
                Shape = (batch, 3, 3)
    """
    pi_over_180=0.017453292519943295
    a, b, c = unit_cell_dimensions[:,:3].T
    cosal, cosbe, cosga = torch.cos(unit_cell_dimensions[:,3:]*pi_over_180).T
    sinal, sinbe = torch.sin(unit_cell_dimensions[:,3:-1]*pi_over_180).T
    # Sometimes rounding errors cause |values| slightly > 1.
    val = torch.clamp((cosal * cosbe - cosga) / (sinal * sinbe), min=-1, max=1)

    gamma_star = torch.arccos(val)
    zeros = torch.zeros_like(a)
    v_a = torch.stack([a * sinbe, zeros, a*cosbe]).T
    v_b = torch.stack([-b * sinal * torch.cos(gamma_star),
                    b*sinal * torch.sin(gamma_star),
                    b*cosal]).T
    v_c = torch.stack([zeros, zeros, c]).T

    matrix = torch.stack([v_a,v_b,v_c], dim=2)

    # Unit cells are valid if cell volume > 0
    # The cell volume is |det(M)|, but don't need the absolute value here
    volume = torch.linalg.det(matrix)
    valid = volume != 0

    return matrix, valid


def get_recip_lattice_metric_tensor(recip_lattice_matrix):
    """Calculate the reciprocal lattice metric tensor

    Args:
        recip_lattice_matrix (tensor):  Reciprocal lattice matrix
                                        Shape = (batch, 3, 3)

    Returns:
        tensor: Reciprocal lattice metric tensor
                Shape = (batch, 3, 3)
    """
    return recip_lattice_matrix @ recip_lattice_matrix.permute(0,2,1)


def get_d_spacing(recip_latt_metric_tensor,hkl):
    """Calculate the d-spacings for the reflections from the Miller indices and
    the reciprocal lattice metric tensor

    Args:
        recip_latt_metric_tensor (tensor):  Reciprocal lattice metric tensor
                                            Shape = (batch, 3, 3)
        hkl (tensor):   Miller indices
                        Shape = (batch, number of reflections, 3)

    Returns:
        tensor: d-spacing for each of the reflections
                Shape = (batch, number of reflections)
    """
    one_over_d_squared = torch.einsum("bij,bji->bi",hkl,torch.einsum(
                                "bij,bkj->bik",recip_latt_metric_tensor,hkl))
    d = 1/torch.sqrt(one_over_d_squared)
    return d


def d_to_tt(d,wavelength=1.54056):
    """Convert d-spacings to twotheta values (in degrees)

    Args:
        d (tensor): d-spacings for each of the reflections
                    Shape = (batch, number of reflections)
        wavelength (float): The wavelength of the radiation. Defaults to
                            1.54056 which is copper K-a1

    Returns:
        tensor: twotheta values for each of the reflections
                Shape = (batch, number of reflections)
    """
    two_times_180_over_pi = 114.59155902616465
    tt = two_times_180_over_pi*torch.arcsin(wavelength/(2*d))
    return tt


def get_zero_point_error(batchsize, device, dtype, zpemin=0.03, zpemax=0.03):
    """
    Generate a random zero-point error to be applied to the peak positions

    Args:
        batchsize (int): Size of the batch dimension
        device (torch.device): Device to generate the tensor on
        dtype (torch.dtype): dtype to use for the ZPEs
        zpemin (float, optional): Lower bound for zero point error in degrees
        zpemax (float, optional): Upper bound for zero point error in degrees

    Returns:
        tensor: Zero point error to be applied to the peak positions
                Shape = (batch, 1)
    """
    zero_point_error = (torch.rand((batchsize,1), device=device, dtype=dtype)
                        * (zpemax - zpemin)) + zpemin
    return zero_point_error