from fume import Fume3dLayer, calculate_fundamental_matrix
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import cpu_project_volume

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

if __name__ == '__main__':
    # print nicer
    np.set_printoptions(suppress=True, precision=6)

    # define cubes in standard sized volume
    cubes_volume = np.zeros((512, 512, 512))
    cubes_volume[20:120, 20:70, 20:70] = 1       # 100 x 50 x 50
    cubes_volume[230:280, 230:330, 230:280] = 1  # 50 x 100 x 50
    cubes_volume[20:70, 390:440, 320:400] = 1    # 50 x 50 x 100

    # load projection matrices
    P1 = np.array([[-1.491806664, -6.0116547806, -0.0032140855, 497.8171012864],
                   [-0.7840045398, 0.0883435020, 6.1466040404, 486.4775230631],
                   [-0.0015992436, 0.0001893259, 0.0000025475, 1.0000000000]])  # <M0>
    P2 = np.array([[5.928340124, -1.7884644150, -0.0147336253, 483.7053684109],
                   [-0.1191121191, -0.7833319327, 6.1436160514, 493.1060850290],
                   [-0.0002700307, -0.0015866526, -0.0000015344, 1.0000000000]])  # <M179>

    # adapt matrices to volume indices instead of mm
    P1 = P1 @ xzy_from_iso @ iso_from_ijk
    P2 = P2 @ xzy_from_iso @ iso_from_ijk

    # debug print
    print("F21", calculate_fundamental_matrix(P_src=P1, P_dst=P2))

    # calculate fundamental matrix to map coordinates from P1 to lines in P2 (l2 = F21 @ x1)
    F21 = torch.tensor(calculate_fundamental_matrix(P_src=P1, P_dst=P2).reshape((1, 3, 3)), device='cuda')
    F12 = torch.tensor(calculate_fundamental_matrix(P_src=P2, P_dst=P1).reshape((1, 3, 3)), device='cuda')

    # create projection images
    view1 = cpu_project_volume(cubes_volume, P1)
    view2 = cpu_project_volume(cubes_volume, P2)

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
    plt.imshow(np.squeeze(CM1.cpu()).T, 'jet', alpha=0.4)
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(view2.T, 'gray')
    plt.imshow(np.squeeze(CM2.cpu()).T, 'jet', alpha=0.4)
    plt.axis('off')
    plt.show()