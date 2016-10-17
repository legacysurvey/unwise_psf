import fitsio
import numpy as np
import fitsio
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import os

from unwise_psf import get_unwise_psf

def all_coadd_id():
    atlas = fitsio.read('~/unwise/pro/allsky-atlas.fits')

    for i, row in enumerate(atlas):
        print i
        rot = get_unwise_psf(None, str(row['coadd_id']))
        print rot.shape

def write_rot_plots(indstart, nproc, out_basedir='/scratch1/scratchdirs/ameisner/unwise_psf_plots', band=1):
    atlas = fitsio.read('~/unwise/pro/allsky-atlas.fits')

    ntile = len(atlas)
    indend = min(indstart + nproc, ntile)

    atlas = atlas[indstart:indend]

    for i, row in enumerate(atlas):
        coadd_id = str(row['coadd_id'])
        print i, coadd_id
        rot = get_unwise_psf(band, coadd_id)
        plt.imshow(rot, cmap='gray', interpolation='nearest', norm=LogNorm(vmin=0.1, vmax=1000000), origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title('W'+str(band)+', '+coadd_id)
        outdir = out_basedir + '/' + coadd_id[0:3]
        outname = outdir + '/' + coadd_id + '.png'
        if not os.path.exists(outdir): os.mkdir(outdir)
        plt.savefig(outname, dpi=250, bbox_inches='tight')
        plt.cla()

def every_tenth_psf(band):
    ntile = 18240

    ind = np.arange(0, 18240, 10)

    for _, i in enumerate(ind):
        print str(_+1) + ' of ' + str(len(ind))
        write_rot_plots(i, 1, out_basedir='/scratch1/scratchdirs/ameisner/test'+str(band), band=band)
