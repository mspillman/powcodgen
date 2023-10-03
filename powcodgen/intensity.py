import torch

def get_MD_PO_components(hkl, reciprocal_lattice_metric_tensor, dspacing, factor_std=0.1):
    """Calculate the terms needed in the March-Dollase preferred orientation
    correction.

    Args:
        hkl (tensor):   The Miller indices of the reflections.
                        Shape = (batch, number of reflections, 3)
        reciprocal_lattice_matrix (tensor): The reciprocal lattice matrix
                                            Shape = (batch, 3, 3)
        factor_std (float, optional):   The standard deviation for the normally
                                        distributed March-Dollase factors.
                                        Defaults to 0.1.

    Returns:
        tuple: Tuple of tensors containing the terms needed in the March-Dollase
        preferred orientation correction function. cosP, sinP = cosine and sine
        of the angle between the Miller indices and the PO axis. Factor = the
        March Dollase factors, PO_axis = the preferred orientation axis.
    """
    batchsize = hkl.shape[0]
    device = hkl.device
    dtype = hkl.dtype
    # Randomly assign the PO axis to be either [1,0,0], [0,1,0] or [0,0,1]
    PO_axis = torch.zeros((batchsize, 3), device=device, dtype=dtype)
    PO_axis_select = torch.randint(0,3,(batchsize,),device=device)
    PO_axis[torch.arange(batchsize, device=device),PO_axis_select] = 1.0


    """u = hkl / torch.sqrt(torch.einsum("bkj,bkj->bk", hkl,
                        torch.einsum("bij,bkj->bki",
                        reciprocal_lattice_matrix,hkl))).unsqueeze(2)"""
    # Dividing the Miller indices by the reciprocal lattice vector lengths is
    # equivalent to multiplying them by the d-spacings
    u = hkl * dspacing.unsqueeze(2)

    cosP = torch.einsum("bij,bj->bi", u, torch.einsum("bij,bj->bi",
                    reciprocal_lattice_metric_tensor, PO_axis))
    one_minus_cosP_sqd = 1.0-cosP**2
    one_minus_cosP_sqd[one_minus_cosP_sqd < 0.] *= 0.
    sinP = torch.sqrt(one_minus_cosP_sqd)

    # MD factor = 1 means no PO. Use a normal distribution with std given in the
    # argument centred at 1.
    factor = 1 + (torch.randn((batchsize,1),device=device,dtype=dtype) * factor_std)

    return cosP, sinP, factor, PO_axis

def apply_MD_PO_correction(intensities, cosP, sinP, factor):
    """Modifies the intensities to account for preferred orientation effects
    using the method of March and Dollase.

    Args:
        intensities (tensor):   Original intensities for the reflections.
            Shape = (batch, number of reflections)
        cosP (tensor):  Cosine of the angle between the Miller indices and the
            preferred orientation axis. Calculated in get_MD_PO_components
            Shape = (batch, number of reflections)
        sinP (tensor):  Sine of the angle between the Miller indices and the
            preferred orientation axis. Calculated in get_MD_PO_components
            Shape = (batch, number of reflections)
        factor (tensor): The March-Dollase factors. Shape = (batch,1)

    Returns:
        tensor: Modified intensities for the reflections given preferred
        orientation. Shape = (batch, number of reflections)
    """
    A_all = (1.0/torch.sqrt(((factor)*cosP)**2+sinP**2/(factor)))**3
    return intensities * A_all