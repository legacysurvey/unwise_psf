import unwise_psf
import fitsio

def test_frames():
    frames = fitsio.read('/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth/000/0000m016/unwise-0000m016-w1-frames.fits')
    psf = unwise_psf.get_unwise_psf(1, '0000m016', frames=frames)

    return psf

def test_tr_w2():

    # http://legacysurvey.org/viewer?ra=163.9380&dec=33.5105&zoom=10&layer=unwise-neo3

    frames = fitsio.read('/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4/e008/163/1637p333/unwise-1637p333-w2-frames.fits')

    psf = unwise_psf.get_unwise_psf(2, '0000m016', frames=frames)

    return psf

def test_poles():

    # http://legacysurvey.org/viewer?ra=97.7563&dec=-66.8662&zoom=8&layer=unwise-neo3

    # what coadd_id does this correspond to ?
    frames = fitsio.read('/global/projecta/projectdirs/cosmo/work/wise/outputs/merge/neo4/fulldepth/096/0964m667/unwise-0964m667-w2-frames.fits')

    psf = unwise_psf.get_unwise_psf(2, '0964m667', frames=frames)

    return psf
