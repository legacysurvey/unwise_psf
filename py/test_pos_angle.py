import fitsio
import numpy as np
import fitsio

from unwise_psf import pos_angle_ecliptic

def all_pos_angles():
    atlas = fitsio.read('~/unwise/pro/allsky-atlas.fits')

    pos_angles = np.zeros(len(atlas))
    for i, row in enumerate(atlas):
        print i
        pos_angles[i] = pos_angle_ecliptic(str(row['coadd_id']))

    return pos_angles
