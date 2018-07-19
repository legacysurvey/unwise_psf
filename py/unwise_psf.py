from astrometry.util.resample import _lanczos_interpolate
import numpy as np
import fitsio
from astrometry.util.starutil_numpy import radectoecliptic
from astrometry.util.starutil_numpy import ecliptictoradec
import os
import warnings

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

    laccs = [np.zeros(nn, np.float32) for im in limages]
    _lanczos_interpolate(L, ixi, iyi, dx, dy, laccs, limages, table=True)

    return (laccs[0]).reshape(sh)

def rotate_psf(psf_image, theta):

    sh = psf_image.shape

    assert(len(sh) == 2) # 2d PSF image
    assert(sh[0] == sh[1]) # square

    sidelen = sh[0]

    half = sidelen/2
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

    bad = (xbox_rot < 0) | (ybox_rot < 0) | (xbox_rot > (sidelen-1)) | (ybox_rot > (sidelen-1))

    psf_rot = do_lanczos_interpolation(psf_image, xbox_rot, ybox_rot)

    psf_rot *= (~bad)

    return psf_rot

def get_astrom_atlas():
    # eventually be smarter about this, with caching

    atlas_fname = os.path.join(os.getenv('WISE_PSF_DIR'), 'astrom-atlas.fits')
    atlas = fitsio.read(atlas_fname)

    return atlas

def _get_astrometry(coadd_id):
    atlas = get_astrom_atlas()
    return (atlas[atlas['COADD_ID'] == coadd_id])[0]

def pos_angle_ecliptic(coadd_id):
    # not intended to be vectorized, coadd_id input should be scalar

    astr = _get_astrometry(coadd_id)

    xcen = astr['CRVAL'][0] - 1.0 # shouldn't be needed
    ycen = astr['CRVAL'][1] - 1.0 # shouldn't be needed

    racen = astr['CRVAL'][0] # deg
    deccen = astr['CRVAL'][1] # deg

    # convert (racen, deccen) to ecliptic
    # input to radectoecliptic should be in degrees
    lambda_cen, beta_cen = radectoecliptic(racen, deccen)

    epsilon = 10*2.75/3600.0 # degrees, 10 pixels

    beta_test = beta_cen + epsilon
    ra_test, dec_test = ecliptictoradec(lambda_cen, beta_test)

    #dx = x_test - xcen
    #dy = y_test - ycen

    radeg = 180.0/np.pi
    scale = 3600.0*radeg/2.75
    dx, dy = ad2xy_tan(ra_test/radeg, dec_test/radeg, racen/radeg, deccen/radeg, scale)

    theta = np.arctan2(-dx, dy)
    return theta*radeg

def ad2xy_tan(ra, dec, ra0, dec0, scale):
    """
    Converts from (lon, lat) to (x, y) based on Tan WCS

    Inputs:
        ra    - array of RA coordinates, assumed radians J2000
        dec   - array of DEC coordinates, assumed radians J2000
        ra0   - array of central RA of projection, radians
        dec0  - array of central DEC of projection, radians
        scale - scale factor of projection in pixels/radian

    Outputs:
        x     - array of x coordinates within each tile, relative to center
        y     - array of y coordinates within each tile, relative to center

    Comments:
       This is my port of an IDL routine, issa_proj_gnom.pro,
       originally written by Doug Finkbeiner.
    """

    A = np.cos(dec)*np.cos(ra-ra0)
    F = scale/(np.sin(dec0)*np.sin(dec) + A*np.cos(dec0))

    x = -F*np.cos(dec)*np.sin(ra-ra0)
    y = F*(np.cos(dec0)*np.sin(dec) - A*np.sin(dec0))

    return x, y

def average_two_scandirs(psf_model):
    # average the two scan directions (not meant to be correct near ecl poles)
    return (psf_model + psf_model[::-1, ::-1])/2.0

def pad_psf_model(model):
    sh = model.shape
    # assume square
    assert(len(sh) == 2)
    assert(sh[0] == sh[1])

    new_sidelen = int(np.ceil(np.sqrt(2)*float(sh[0])))
    if (new_sidelen % 2) == 0:
        new_sidelen += 1

    psf_padded = np.zeros((new_sidelen, new_sidelen))

    # embed the nonzero portion of the PSF model within the padded cutout
    ind_l = (new_sidelen - sh[0])/2

    psf_padded[ind_l:(ind_l+sh[0]), ind_l:(ind_l+sh[0])] =  model

    return psf_padded

def get_unwise_psf(band, coadd_id, sidelen=None, pad=False):
    
    assert(band <= 4)
    assert(band >= 1)

    # i like having centroid be at the center of a pixel rather than corner
    if sidelen is not None:
        assert(sidelen % 2)

    # read in the PSF model file
    model = fitsio.read(os.path.join(os.getenv('WISE_PSF_DIR'), 'psf_model_w'+str(band)+'.fits'))
    model = average_two_scandirs(model)

    if pad:
        model = pad_psf_model(model)

    # figure out rotation angle
    theta = pos_angle_ecliptic(coadd_id)

    # rotate with rotate_psf
    rot = rotate_psf(model, theta)

    if sidelen is not None:
        sh = (rot.shape)
        if (sidelen > sh[0]):
            warnings.warn('requested sidelength is larger than that of the PSF model')
        else:
            half = sh[0]/2
            rot = rot[(half - sidelen/2):(half + sidelen/2 + 1), (half - sidelen/2):(half + sidelen/2 + 1)]

    return rot
