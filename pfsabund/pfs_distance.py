"""

This file contains code to estimate distances 
from photometry and Gaia parallax, if available. 

"""


import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia


def estimate_dist(ra, dec, mag, filter):

    print("Estimate distance for an object at (RA, DEC) = (%.7f, %.6f)"\
          %(ra, dec))

    # Query parallax 
    gaiaid, parallax, parallax_error = query_GaiaEDR3(ra, dec, mag, filter)

    if np.isscalar(parallax) == False:
        print("Multiple Gaia objects found at (RA, DEC) = (%.7f, %.6f)"\
              %(ra, dec))
        
    elif gaiaid == "":
        print("No objects found at (RA, DEC) = (%.7f, %.6f)"\
              %(ra, dec))
        

    
    # Get probabiliity distribution for distance
    dist, disterr = from_parallax(parallax, parallax_error)

    # Calculate distance modulus and its error 
    #dm, dmerr = calc_dmod_from_distances(dist, disterr)
    
    
    return(gaiaid, dist, disterr)



def query_GaiaEDR3(ra, dec, mag, filter, radius = 60.):

    G_guess = guess_Gaia_mag_from_color(mag, filter)

    
    coord = SkyCoord(ra = ra, dec = dec, unit = (u.degree, u.degree), \
                     frame = 'icrs')
    
    radius = u.Quantity(radius/3600., u.deg)
    result = Gaia.query_object(coordinate = coord, radius = radius)


    Gmag0 = result["phot_g_mean_mag"]
    filt = np.abs(Gmag0 - G_guess)<1.0
    parallax0 = result["parallax"][filt]
    parallax_error0 = result["parallax_error"][filt]
    gaiaid0 = result["source_id"][filt]


    
    if len(parallax0) == 0:
        gaiaid = ""
        parallax = np.nan
        parallax_error = np.nan
    elif len(parallax0) >1:
        gaiaid = gaiaid0
        parallax = parallax0
        parallax_error = parallax_error0
        
    else:
        gaiaid = gaiaid0[0]
        parallax = parallax0[0]
        parallax_error = parallax_error0[0]
        
    return(gaiaid, parallax, parallax_error)


def guess_Gaia_mag_from_color(mag, filter):


    # G - g vs (g-i) calibration
    a = [-0.074189, -0.51409, -0.080607, 0.0016001, 0.046848]
    asig = 0.046848

    
    # G - g vs (g-r) calibration
    b = [-0.038025, -0.76988, -0.1931, 0.0060376, 0.065837]
    bsig = 0.065837
    

    
    if filter[0] != "g" or filter[1] != "r" or filter[2] != "i":
        G_guess = np.nan
        return()
        
    else:
        gi = mag[0] - mag[2]
        
        gr = mag[0] - mag[1]


        
        G_guess_gi = a[0] + a[1] * gi + a[2] * gi**2 + a[3] * gi**3 + mag[0]
        
        G_guess_gr = b[0] + b[1] * gr + b[2] * gr**2 + b[3] * gr**3 + mag[0]

        w = (1./asig**2 + 1./bsig**2)
        G_guess = (G_guess_gi/asig**2 + G_guess_gr/bsig**2)/w

        
    return(G_guess)


def from_parallax(parallax, parallax_error, prior = False):

    # This is very preliminary and should be taken with caution.
    # See https://arxiv.org/pdf/1804.10121.pdf

    # parallax should be in [mas]
    # distance will be in [kpc]

    if parallax > 0.0:
        dist = 1. / parallax
        disterr = 1. * parallax_error / parallax**2
    else:
        dist = np.nan
        disterr = np.nan

    return(dist, disterr)


def calc_dmod_from_distances(dist, disterr):

    dpc = dist * 1000.
    dpcerr = disterr * 1000.
    
    dm = 5.*(np.log10(dpc) - 1.)
    ddm = (5./np.log(10.)) * dpcerr/dpc

    return(dm, ddm)


#ra = 226.720833
#dec = 30.011046999999998
#mag = np.array([9.68638, 8.49738, 8.02438, 7.80738, 7.70438])
#filter = np.array(['g', 'r', 'i', 'z', 'y'])


#dm, dmerr = estimate_dmod(ra, dec, mag, filter)
#print(dm, dmerr)

