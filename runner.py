import numpy as np
import matplotlib.pyplot as plt
import fluxfm



if __name__ == '__main__':
    print('XFM h5 Processing')
    print('Have you activated your conda profile?')


    """
    Constants:
    """
    EIGER_nx = 1062
    EIGER_ny = 1028
    MAX_PX_COUNT = 2 ** 32 - 1
    MASK_MAX = 1e11
    cam_length = 0.692
    wavelength = 0.67018E-10
    pix_size = 75E-6
    image_center = (514.375,532.340)


    """
    Mask details:
    """
    # xmin, xmax, ymin, ymax
    bboxes =  [(501.0, 525.0, 601.0, 619.0),
                (0.0, 1028.0, 505.0, 575.0),
                (501.0, 525.0, 305.0, 478.0),
                (540.0, 561.0, 308.0, 349.0),
                (524.0, 570.0, 440.0, 480.0),
                (511.0, 515.0, 512.0, 1062.0)] 
    circs = [(image_center[0],image_center[1], 100.0)]


    """
    Variables:    
    """
    dset = fluxfm.XfmHfiveDataset()
    dset.group = 'vortmo'
    dset.tag = '66269_80'
    dset.bboxes = bboxes
    dset.circs = circs
    dset.image_center = image_center
    dset.cam_length = cam_length
    dset.wavelength = wavelength
    dset.pix_size = pix_size
    dset.dpath = f'/data/xfm/data/2021r1/Binns_16777/raw/eiger/{dset.group}/{dset.tag}/'
    dset.apath = f'/data/xfm/data/2021r1/Binns_16777/analysis/eiger/SAXS/{dset.group}/{dset.tag}/'
    dset.grab_dset_members()
    dset.mk_scratch('')
    """
    Processing:
    """
    # Sum the 2D images first:

    # dset.dsum = dset.sum_h5s(dump=True)                    # Generate sum file
    dset.dsum = np.load(dset.scratch+dset.tag + '_sum.npy')  # Read in premade sum file (much quicker)

    # Generate or inspect the mask:
    # dset.mask = dset.gen_mask(dset.dsum,bboxes=True,circs=True,max_lim = MASK_MAX,dump=True) # Generate mask file
    dset.mask = np.load(dset.scratch+dset.tag + '_mask.npy') # Read in premade mask
    dset.inspect_mask(dset.dsum)
    dset.inspect_unwarp(dset.dsum * dset.mask)

    

    # Once you're happy with the mask
    # separate the h5 files into hits
    # Caution: takes a long time!

    # Atomise the h5 files into dbins:
    # dset.atomize_h5(folder='h5_frames', limit=10, masked=True) # dbins are gnerated in the folder with a manifest.txt file for py3padf

    # Atomise the h5 files in 1D npy:
    # Reduces to either q ('q') or s ('s'), will implement 2theta
    # dset.reduce_h5(scatter_mode = 'q')

    
