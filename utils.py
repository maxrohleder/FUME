import numpy as np


def project_points(points, pmat):
    assert points.ndim == 2 and points.shape[1] == 3
    S_ijk = np.pad(points, ((0, 0), (0, 1)), constant_values=1)  # make homogenous

    # project coordinates onto detector
    samples_dec_homog = pmat @ S_ijk.T
    samples_dec = samples_dec_homog[:2] / samples_dec_homog[-1]
    return samples_dec

def cpu_project_points_on_image(points, pmat, detector_shape=(976, 976)):
    """
    simple forward projection method which calculates the detector positions for all voxels == 1 in the given volume.
    @param points: points in shape (n, 3)
    @param pmat:
    @return:
    """
    samples_dec = project_points(points, pmat)

    # create empty image
    dec_img = np.zeros(detector_shape)

    # check boundaries and round to pixels
    samples_dec_boundary_checked = samples_dec[:, np.all((samples_dec < detector_shape[0]-1) * (samples_dec > 0), axis=0)]
    outside_ratio = (samples_dec.shape[1] - samples_dec_boundary_checked.shape[1]) / samples_dec.shape[1]
    positions = np.round(samples_dec_boundary_checked).astype(int)  # (2, 50000)

    # create detector image by accumulating samples on detector
    pos_u, counts = np.unique(positions, axis=1, return_counts=True)
    dec_img[pos_u[0], pos_u[1]] = counts
    return dec_img

def cpu_project_volume(volume, pmat, detector_shape=(976, 976)):
    """
    simple forward projection method which calculates the detector positions for all voxels == 1 in the given volume.
    @param volume: binary volume. only voxels == 1 will be projected. (for performance reasons)
    @param pmat:
    @return:
    """
    # prepare matrices and output
    S_ijk = np.array(np.where(volume >= 1)).T
    return cpu_project_points_on_image(S_ijk, pmat, detector_shape=detector_shape)