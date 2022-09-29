from fume import Fume3dLayer, calculate_fundamental_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging

v_mm = 0.313
xzy_from_iso = np.array([[v_mm, 0, 0, 0],
                         [0, v_mm, 0, 0],
                         [0, 0, v_mm, 0],
                         [0, 0, 0, 1]])
c = 255.5  # (512 / 2) - 0.5
iso_from_ijk = np.array([[1, 0, 0, -c],
                         [0, 1, 0, -c],
                         [0, 0, 1, -c],
                         [0, 0, 0, 1]])


def cpu_project_volume(volume, pmat, detector_shape=(976, 976), block_filter=False):
    """
    simple forward projection method which calculates the detector positions for all voxels == 1 in the given volume.
    :param volume: binary volume. only voxels == 1 will be projected. (for performance reasons)
    :param pmat: projection matrix in shape (3, 4)
    :param detector_shape: preferred detector resulution (must fit to projection matrices)
    :param block_filter: if True, a block filter is applied to smooth the output
    :return: detector image
    """
    # extract coordinates and make homogenous
    S_ijk = np.array(np.where(volume >= 1)).T
    S_ijk = np.pad(S_ijk, ((0, 0), (0, 1)), constant_values=1)

    # project coordinates onto detector
    samples_dec_homog = pmat @ S_ijk.T
    samples_dec = samples_dec_homog[:2] / samples_dec_homog[-1]

    # create empty image
    dec_img = np.zeros(detector_shape)

    # check boundaries and round to pixels
    samples_dec_boundary_checked = samples_dec[:, np.all((samples_dec < detector_shape[0] - 1) * (samples_dec > 0), axis=0)]
    outside_ratio = (samples_dec.shape[1] - samples_dec_boundary_checked.shape[1]) / samples_dec.shape[1]
    positions = np.round(samples_dec_boundary_checked).astype(int)  # (2, n)

    # maybe warn if a lot of projectiona do not land on detector
    if outside_ratio > 0.2:
        logging.warning(f"{outside_ratio * 100} percent ouf points not projected onto detector")

    # create detector image by accumulating samples on detector
    pos_u, counts = np.unique(positions, axis=1, return_counts=True)
    dec_img[pos_u[0], pos_u[1]] = counts

    # as point-wise projection introduces artefacts, optionally smooth output image
    if block_filter:
        dec_img = np.squeeze(
            torch.conv2d(
                torch.tensor(dec_img.reshape(1, *dec_img.shape), dtype=torch.float),
                torch.ones((1, 1, 5, 5), dtype=torch.float),
                padding=2).numpy()
        )

    return dec_img.astype(np.float64)


if __name__ == '__main__':
    # print nicer
    np.set_printoptions(suppress=True, precision=6)

    # define cubes in standard sized volume
    cubes_volume = np.zeros((512, 512, 512))
    cubes_volume[20:120, 20:70, 20:70] = 1       # 100 x 50 x 50
    cubes_volume[230:280, 230:330, 230:280] = 1  # 50 x 100 x 50
    cubes_volume[20:70, 390:440, 320:400] = 1    # 50 x 50 x 100

    # load projection matrices (from a real 3d scan, views roughly 90 degrees apart)
    P1 = np.array([[-1.491806664, -6.0116547806, -0.0032140855, 497.8171012864],
                   [-0.7840045398, 0.0883435020, 6.1466040404, 486.4775230631],
                   [-0.0015992436, 0.0001893259, 0.0000025475, 1.0000000000]])  # <M0>
    P2 = np.array([[5.928340124, -1.7884644150, -0.0147336253, 483.7053684109],
                   [-0.1191121191, -0.7833319327, 6.1436160514, 493.1060850290],
                   [-0.0002700307, -0.0015866526, -0.0000015344, 1.0000000000]])  # <M179>

    # adapt matrices to volume indices instead of mm
    P1 = P1 @ xzy_from_iso @ iso_from_ijk
    P2 = P2 @ xzy_from_iso @ iso_from_ijk

    # calculate fundamental matrix to map coordinates from P1 to lines in P2 (l2 = F21 @ x1)
    F21 = torch.tensor(calculate_fundamental_matrix(P_src=P1, P_dst=P2).reshape((1, 3, 3)), device='cuda')
    F12 = torch.tensor(calculate_fundamental_matrix(P_src=P2, P_dst=P1).reshape((1, 3, 3)), device='cuda')

    # create projection images
    view1 = cpu_project_volume(cubes_volume, P1, block_filter=True)
    view2 = cpu_project_volume(cubes_volume, P2, block_filter=True)

    # binarize projections and convert to tensor
    view1_bin, view2_bin = np.zeros_like(view1), np.zeros_like(view2)
    view1_bin[view1 > 0], view2_bin[view2 > 0] = 1, 1
    view1_bin = torch.tensor(view1_bin.reshape((1, 1, 976, 976)), device='cuda')
    view2_bin = torch.tensor(view2_bin.reshape((1, 1, 976, 976)), device='cuda')

    # translate
    fume3d = Fume3dLayer()
    CM1 = fume3d(view2_bin, F21, F12)
    CM2 = fume3d(view1_bin, F12, F21)

    # plot
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(view1.T, 'gray')
    plt.imshow(np.squeeze(CM1.cpu()).T, 'Blues', alpha=0.4)
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(view2.T, 'Blues')
    plt.imshow(np.squeeze(CM2.cpu()).T, 'gray', alpha=0.4)
    plt.axis('off')
    plt.show()
