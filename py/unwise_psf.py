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

def rotate_using_frames(model, frames, oversample=2):
    assert(np.sum(frames['included']) != 0)

    frames = frames[frames['included'] != 0]

    # i suppose there could be rare cases where PA actually is zero
    assert(np.sum(frames['pa'] == 0) == 0)

    # concept: instead of rotating thousands of times, render the PSF at high
    # resolution (~2-4x oversampled?) in polar coordinates.  Then convolve
    # in 1D to get the rotated and summed PSF.  Finally, transform back to
    # cartesian coordinates.

    order = 4
    ntheta = 4*model.shape[0]*oversample
    nrad = model.shape[0]*oversample
    szo2 = model.shape[0] // 2
    rr = np.linspace(-5, szo2*np.sqrt(2), nrad)
    tt = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    xx = rr[:, None]*np.cos(tt[None, :])+szo2
    yy = rr[:, None]*np.sin(tt[None, :])+szo2
    from scipy.ndimage import map_coordinates
    modelpolar = map_coordinates(model, [xx, yy], order=order,
                                 mode='constant', cval=0.,
                                 output=np.dtype('f4'))

    tpix = (360-(frames['pa'].astype('f8') % 360))*ntheta/360
    from numpy import fft
    modelpolar = fft.rfft(modelpolar, axis=1)
    freq = fft.rfftfreq(ntheta)
    # this next line is 75% of time at N_frames = 30k.  Most above here could
    # be saved and reused.  We could speed this up dramatically in the many
    # frame case by doing an FFT of the PA distribution rather
    # than explicitly calculating this.  I preferred the explicit calculation
    # below to make sure to get the angle exactly right (not just right up to
    # one pixel at the far edge of the stamp), but one could imagine doing an
    # FFT plus one phase offset to get the mean exactly right.
    convarr = np.sum(np.exp(-2*np.pi*1j*(tpix[None, :]*freq[:, None])),
                     axis=1)
    convarr /= len(frames)
    modelpolar = fft.irfft(modelpolar*convarr, axis=1)
    szo2 = model.shape[0] // 2
    xo, yo = np.mgrid[-szo2:szo2+1, -szo2:szo2+1]
    ro = np.sqrt(xo**2 + yo**2)
    to = np.arctan2(yo, xo) % (2*np.pi)
    ro = np.interp(ro, rr, np.arange(len(rr)).astype('f4'))
    to = np.interp(to, tt, np.arange(len(tt)).astype('f4'))
    res = map_coordinates(modelpolar, [ro, to], order=order,
                          mode='wrap', output=np.dtype('f4'))
    return res

def get_unwise_psf(band, coadd_id, sidelen=None, pad=False, frames=None):
    
    assert(band <= 4)
    assert(band >= 1)

    # i like having centroid be at the center of a pixel rather than corner
    if sidelen is not None:
        assert(sidelen % 2)

    # read in the PSF model file
    model = fitsio.read(os.path.join(os.getenv('WISE_PSF_DIR'), 'psf_model_w'+str(band)+'.fits'))

    if frames is None:
        model = average_two_scandirs(model)

    if pad:
        model = pad_psf_model(model)

    if frames is None:
        # figure out rotation angle
        theta = pos_angle_ecliptic(coadd_id)
        # rotate with rotate_psf
        rot = rotate_psf(model, theta)
    else:
        rot = rotate_using_frames(model, frames)

    if sidelen is not None:
        sh = (rot.shape)
        if (sidelen > sh[0]):
            warnings.warn('requested sidelength is larger than that of the PSF model')
        else:
            half = sh[0]/2
            rot = rot[(half - sidelen/2):(half + sidelen/2 + 1), (half - sidelen/2):(half + sidelen/2 + 1)]

    return rot
