from astrometry.util.resample import _lanczos_interpolate
import numpy as np
from astrometry.util.util import lanczos3_interpolate

# always assume odd sidelengths (?)
# always assume square PSF image (?)

def do_lanczos_interpolation(image, x_interp, y_interp):

    sh = x_interp.shape # will use this to reshape the output interpolated vals

    L = 3 # hardwired for now
    ixi = np.ravel(np.round(x_interp))
    iyi = np.ravel(np.round(y_interp))

    dx = (np.ravel(x_interp) - ixi).astype(np.float32)
    dy = (np.ravel(y_interp) - iyi).astype(np.float32)

    ixi = ixi.astype(np.int32)
    iyi = iyi.astype(np.int32)

    limages = [image]
    nn = len(ixi) # ??

    print ixi.dtype

    laccs = [np.zeros(nn, np.float32) for im in limages]
    _lanczos_interpolate(L, ixi, iyi, dx, dy, laccs, limages, table=True)
    # should probably reshape the output
    # to be an image

    return (laccs[0]).reshape(sh)

def rotate_psf(psf_image, theta):

    sh = psf_image.shape

    assert(len(sh) == 2) # 2d PSF image
    assert(sh[0] == sh[1]) # square

    sidelen = sh[0]

    half = sidelen/2
    print half
    # assume theta in degrees !!

    xbox = np.arange(sidelen*sidelen).reshape(sidelen, sidelen) % sidelen
    ybox = np.arange(sidelen*sidelen).reshape(sidelen, sidelen) / sidelen

    xbox = xbox.astype(float)
    ybox = ybox.astype(float)

    radeg = 180.0/np.pi
    theta_rad = theta/radeg # in radians
    costheta = np.cos(theta_rad)
    sintheta = np.sin(theta_rad)

    # the conventions here make the rotations E thru N
    xbox_rot = (xbox - half)*costheta + (ybox - half)*sintheta
    ybox_rot = -1*(xbox - half)*sintheta + (ybox - half)*costheta

    xbox_rot += half
    ybox_rot += half

    # need the interpolation command here !!
    psf_rot = do_lanczos_interpolation(psf_image, xbox_rot, ybox_rot)

    return psf_rot
