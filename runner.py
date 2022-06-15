import numpy as np
import matplotlib.pyplot as plt
import fluxfm
import constants_config


if __name__ == '__main__':
    print('XFM h5 Processing')
    dset = fluxfm.XfmHfiveDataset()
    dset.experiment_id = constants_config.experiment_id


    dset.cam_length = constants_config.cam_length
    dset.wavelength = constants_config.wavelength
    dset.pix_size = constants_config.pix_size
    dset.max_lim = constants_config.MAX_PX_COUNT
    dset.dpath = f'{constants_config.beamtime_data_path}{dset.group}/{dset.tag}/'
    dset.apath = f'{constants_config.beamtime_analysis_path}{dset.group}/{dset.tag}/'


    # Here you select the group and run for the data set
    dset.group = 'DLC'
    run_num = 48

    image_center = (514.425, 530.777)

    """
    Mask details:
    """
    # xmin, xmax, ymin, ymax
    bboxes = [  # (501.0, 525.0, 601.0, 619.0),
        (0.0, 1028.0, 505.0, 575.0),
        # (501.0, 525.0, 305.0, 478.0),
        # (540.0, 561.0, 308.0, 349.0),
        # (524.0, 580.0, 375.0, 512.0),
        # (485.0, 535.0, 0.0, 1062.0),
        # (319.0, 375.0, 305.0, 355.0),
        # (252.0, 258.0, 0.0, 1062.0),
        # (770.0, 775.0, 0.0, 1062.0),
        # (0.0, 1062.0, 254.0, 259.0),
        # (0.0, 1062.0, 803.0, 809.0)
    ]
    # Final value in the list is the radius
    circs = [(image_center[0], image_center[1], 0.00)]

    # Set the paths and pass mask values to the XfmHfiveDataset
    dset.tag = f'{constants_config.maia_num + run_num}_{run_num}'
    dset.bboxes = bboxes
    dset.circs = circs
    dset.image_center = image_center
    dset.grab_dset_members()
    dset.mk_scratch('')

    """
    Here we run fluxfm commands
    """
    dset.quick_overview(run_limit=1)  # set to 1 to examine just the first run, quickest method



        #
        # """
        # Processing:
        # """
        #
        # # Grab an overview of the data structure. frame numbers etc.
        # # this will also out put a series of profiles for each h5 (*_sum_h5_k_red.dat/npy)
        # #    Plot these up to compare how the profiles change through the whole data set if
        # #    you have multple h5s
        # # Turn me off once you've done once
        # # dset.overview(run_limit=5)
        # #
        # # dset.run_movie(run_id=3, step=20,cmin=0,cmax=20)
        # #
        #
        # # Here you can check the mask and the centering
        # dset.dsum = np.load(f'{dset.apath}{dset.tag}_sum.npy')
        # dset.gen_mask(dset.dsum, bboxes=True, circs=True, max_lim=constants_config.MASK_MAX, dump=True)
        # dset.mask = np.load(f'{dset.apath}{dset.tag}_mask.npy')
        # # dset.scatter_shot_inspect(sample_size=50)
        # # uw = dset.frm_integration(dset.dsum)
        # # uwai = dset.pyFAI_frm_integration(dset.dsum*dset.mask,unit="q_nm^-1")
        # # np.savetxt(f'{dset.apath}{dset.tag}_sum_qnm.dat', uwai)
        # # uwai = dset.pyFAI_frm_integration(dset.dsum*dset.mask,unit="q_nm^-1")
        # # np.savetxt(f'{dset.apath}{dset.tag}_sum_2th.dat', uwai)
        #
        # # dset.inspect_mask(dset.dsum*dset.mask, cmin=0, cmax=1e8)
        # # dset.inspect_unwarp(dset.dsum * dset.mask,cmin=0,cmax=1e6)
        # plt.figure()
        # uwai = dset.frm_integration(dset.dsum * dset.mask)
        # print(uwai.shape)
        # plt.plot(uwai[:, 0], uwai[:, 1])
        # plt.title('pyFAI integration')
        # plt.show()
        # np.savetxt(f'{dset.apath}{dset.tag}_pyfai_sum_2th.txt', uwai)
        # np.save(f'{dset.apath}{dset.tag}_pyfai_sum_2th.npy', uwai)
        #
        # # Now to prepare for PADF calculations, atomize (using start and limit if you want)
        #
        # # LIMITS ARE GIVEN IN PYTHON COUNTING
        # dset.atomize_h5(folder='h5_frames', start=0, limit=3,
        #                 masked=False)  # dbins are gnerated in the folder with a manifest.txt file for py3padf
        # dset.reduce_h5(scatter_mode='s', start=0, limit=3)
        #
        # #
        # # dset.define_average_profile()
        #
        # # plt.figure()
        # # plt.title("average profile")
        # # plt.plot(dset.average_profile[:,0], dset.average_profile[:,1])
        # # plt.show()
        # # dset.filter_against_average(itera=1, rfac_threshold=1.0, inspect=False, qlims=[0,10])
