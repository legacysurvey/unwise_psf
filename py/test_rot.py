import fitsio
import numpy as np
import fitsio
import matplotlib.pyplot as plt
import os

from unwise_psf import get_unwise_psf

def all_coadd_id():
    atlas = fitsio.read('~/unwise/pro/allsky-atlas.fits')

    for i, row in enumerate(atlas):
        print i
        rot = get_unwise_psf(None, str(row['coadd_id']))
        print rot.shape

def write_rot_plots(indstart, nproc):
    atlas = fitsio.read('~/unwise/pro/allsky-atlas.fits')

    ntile = len(atlas)
    indend = min(indstart + nproc, ntile)

    atlas = atlas[indstart:indend]

    out_basedir = '/scratch1/scratchdirs/ameisner/unwise_psf_plots'

    for i, row in enumerate(atlas):
        coadd_id = str(row['coadd_id'])
        print i, coadd_id
        rot = get_unwise_psf(None, coadd_id)
        plt.imshow(np.log10(np.maximum(rot, 0.1)),interpolation='nearest', cmap='gray', vmin=0.1, vmax=5, origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.title(coadd_id)
        outdir = out_basedir + '/' + coadd_id[0:3]
        outname = outdir + '/' + coadd_id + '.png'
        if not os.path.exists(outdir): os.mkdir(outdir)
        plt.savefig(outname, dpi=250, bbox_inches='tight')
        plt.cla()
