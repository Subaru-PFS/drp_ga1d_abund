"""
@author: I. Escala (Caltech, Princeton)

Parts of PFS 1D Abundance Pipeline, adapted from the spectral synthesis method of
Escala et al. 2019. 

Utilitiy classes for use with MeasurePFSAbund()

"""

from __future__ import absolute_import
from scipy.interpolate import splrep, splev
from pfsabund import smooth_gauss
from pfsabund import pfs_phot as phot
import numpy as np
import pickle
import os
import sys

class PFSUtilities():

    """
    Dictionary class to store information concerning the PFS spectrum and
    derived stellar parameters, such as the abundances.
    """
    
    def __init__(self):
    
        """
        Create and initialize the attributes of the MeasurePFSAbund class
        """
        
    def get_phot(self, pfs=None, dm=0., ddm=0.):
    
        """
        Get quantities inferred from photometry, such as the photometric effective
        temperature and the photometric surface gravity.
        
        Parameters
        ----------
        pfs: PFSObject dictionary
        dm: distance modulus to point source
        ddm: error on the distance modulus to the point source
        """
        
        if (np.isnan(pfs.prop('mag')[pfs.prop('filter') == 'i']) or\
            np.isnan(pfs.prop('mag')[pfs.prop('filter') == 'g'])):
                raise ValueError('Invalid photometric value in either g or i band')
    
        #Use a grid of stellar evolutionary models to infer the photometric
        #quantities via interpolation, or extrapolation as necessary
        phot_dict = phot.from_phot(pfs.prop('mag')[pfs.prop('filter') == 'i'], 
                                   pfs.prop('mag')[pfs.prop('filter') == 'g']-\
                                   pfs.prop('mag')[pfs.prop('filter') == 'i'], 
                                   dm=dm, ddm=ddm)
         
        #select an age to assume for the isochrones in the abundance measurement
        #process
        age_i = 12                        
        wage = np.where(np.round(phot_dict[0]['ages']/1.e9, decimals=0) == age_i)[0][0]

        #teff_phot_dummy = np.full( len(pfs.prop('teffphot')), 4100. )
        #logg_phot_dummy = np.full( len(pfs.prop('loggphot')), 1. )
        #tefferr_phot_dummy = np.full( len(pfs.prop('teffphoterr')), 100.)
        
        pfs.assign(phot_dict[0]['teff'][wage], 'teffphot')
        pfs.assign(phot_dict[0]['logg'][wage], 'loggphot')
        
        teffphoterr = np.nanmax([phot_dict[0]['err_teff'][wage], 100.])
        pfs.assign(teffphoterr, 'teffphoterr')
        
        loggphoterr = np.nanmax([phot_dict[0]['err_logg'][wage], 0.1])
        pfs.assign(loggphoterr, 'loggerr')
        
        return

    def smooth_gauss_wrapper(self, lambda1, spec1, lambda2, dlam_in):
    
        """
        A wrapper around the Fortran routine smooth_gauss.f, which
        interpolates the synthetic spectrum onto the wavelength array of the
        observed spectrum, while smoothing it to the specified resolution of the
        observed spectrum.
        Adapted into Python from IDL (E. Kirby)
    
        Parameters
        ----------
        lambda1: array-like: synthetic spectrum wavelength array
        spec1: array-like: synthetic spectrum normalized flux values
        lambda2: array-like: observed wavelength array
        dlam_in: float, or array-like: full-width half max resolution in Angstroms
                 to smooth the synthetic spectrum to, or the FWHM as a function of wavelength
             
        Returns
        -------
        spec2: array-like: smoothed and interpolated synthetic spectrum, matching observations
        """
    
        if not isinstance(lambda1, np.ndarray): lambda1 = np.array(lambda1)
        if not isinstance(lambda2, np.ndarray): lambda2 = np.array(lambda2)
        if not isinstance(spec1, np.ndarray): spec1 = np.array(spec1)
    
        #Make sure the synthetic spectrum is within the range specified by the
        #observed wavelength array
        n2 = lambda2.size; n1 = lambda1.size
    
        def findex(u, v):
            """
            Return the index, for each point in the synthetic wavelength array, that corresponds
            to the bin it belongs to in the observed spectrum
            e.g., lambda1[i-1] <= lambda2 < lambda1[i] if lambda1 is monotonically increasing
            The minus one changes it such that lambda[i] <= lambda2 < lambda[i+1] for i = 0,n2-2
            in accordance with IDL
            """
            result = np.digitize(u, v)-1
            w = [int((v[i] - u[result[i]])/(u[result[i]+1] - u[result[i]]) + result[i]) for i in range(n2)]
            return np.array(w)
    
        f = findex(lambda1, lambda2)

        #Make it such that smooth_gauss.f takes an array corresponding to the resolution
        #each point of the synthetic spectrum will be smoothed to
        if isinstance(dlam_in, list) or isinstance(dlam_in, np.ndarray): 
            dlam = dlam_in
        else: 
            dlam = np.full(n2, dlam_in)
        
        dlam = np.array(dlam)
    
        dlambda1 = np.diff(lambda1)
        dlambda1 = dlambda1[dlambda1 > 0.]
        halfwindow = int(np.ceil(1.1*5.*dlam.max()/dlambda1.min()))
    
        #Python wrapped fortran implementation of smooth gauss
        spec2 = smooth_gauss.smooth_gauss(lambda1, spec1, lambda2, dlam, f, halfwindow)
    
        return spec2
        
    def calc_breakpoints(self, arr=None, npix=None):
    
        """
        Helper function for slatec_splinefit(). Based on IDL function FIND_BKPT. 
        Calculate the breakpoints for a given array, for use with a B-spline, 
        given a number of pixels to include in each interval.
        """
    
        nbkpts = int(float(len(arr))/npix)
        xspot = np.array(list(range(nbkpts)))*len(arr)/(nbkpts - 1)
        xspot = np.round(xspot).astype(int)
        if len(xspot) > 2: bkpt = arr[xspot[1:-1]]
        else: bkpt = None
    
        return bkpt
        
    def slatec_splinefit(self, x=None, y=None, innvar=None, sigma_l=None, sigma_u=None,
        npix=None, maxiter=None):
        
        """
        Python implementation of the IDL code SLATEC_SPLINEFIT to determine breakpoints
        and spline coefficients. Helper function to continuum_normalize()
        """

        mask = np.full(len(x), True) #initialize pixel mask
       
        bad = np.where(innvar <= 0.)[0] #identify bad pixels
        #and remove them from the pixel mask
        if len(bad) > 0: 
            mask[bad] = False 

        k = 0 #counter for iterations of loop
        while k < maxiter: #while the counter is less than the maximum number of iterations
    
            oldmask = mask #store the initial mask (or the mask from the last iteration)
            
            ##calculate the breakpoints of the spline
            break_points = self.calc_breakpoints(x[mask], npix) 
            #calculate the coefficients of the spline
            tck = splrep(x[mask], y[mask], w=innvar[mask], t=break_points)
            cont = splev(x, tck) #determine the continuum
    
            #calculate the difference between the continuum and the observed spectrum,
            #weighted by the square root of the inverse variance
            diff = (y - cont)*np.sqrt(innvar) 
            #Identify pixels outside of the upper and lower bounds indicated for sigma clipping
            wbad = np.where( (diff < -sigma_l)| (diff > sigma_u) | (np.sqrt(innvar) <= 0.) )[0]
    
            #If no points are sigma clipped, then exit the while loop -- continuum has converged
            if len(wbad) == 0: 
                break
            
            #Otherwise, continue  
            else:
            
                mask = np.full(len(x), True) #initialize a new mask
                mask[wbad] = False #mask out bad pixels
                
                #If the continuum fit has converged, then break out of the while loop
                if np.sum(np.abs(mask.astype(int) - oldmask.astype(int))) == 0:
                    break
                    
                #Otherwise, continue to iterate until the maximum number of iterations
                #is achieved
                else: 
                    k += 1
    
        return break_points, tck

        
    def continuum_normalize(self, pfs=None, wavelength_range=None, sigma_l=0.1, 
        sigma_u=5., npix=200, maxiter=5, k=3):
    
        """
        Perform the initial continuum normalization.
        Find the B-spline represenation of a 1D curve.
        Default assumption k = 3, cubic splines recommended. Even values of k should be 
        avoided, especially with small s values. 1 <= k <= 5.
        
        NOTE: Telluric regions according to Kirby et al. 2008 are ommited, although for
        PFS simulated spectra, this is not strictly necessary. 
        """
        
        #Regions strongly affected by telluric absorption, from Kirby et al. 2008
        tell_mask = np.array([[6864., 6935.], [7591., 7694.], [8938.,9100.]])
        
        #Regions to continuum normalize
        if wavelength_range is None:
            wavelength_range = [pfs.prop('wvl').min(), pfs.prop('wvl').max()]
    
        #Select the regions within the above specified range
        w = np.where( (pfs.prop('wvl') >= wavelength_range[0]) &\
                      (pfs.prop('wvl') <= wavelength_range[1]) )[0]
        
        m = len(pfs.prop('wvl')[w])
        #Check if the number of data points is larger than the degree of the spline
        if m < k:
            sys.stderr.write(f'Fewer data points (m = {m}) than degree of spline (k = {k})\n')
            sys.exit()
        
        else:
        
            #Initialize the continuum mask
            mask = np.full(len(pfs.prop('wvl')[w]), True, dtype=bool)
            
            #Exclude problematic pixels from the continuum normalization
            mask[ np.where( (pfs.prop('flux')[w] < 0.) & (pfs.prop('ivar')[w] <= 0.) &\
                           (~np.isfinite(pfs.prop('ivar')[w])) &\
                           (np.isnan(pfs.prop('flux')[w])) ) ] = False
                           
            #Mask the telluric regions
            for tell in tell_mask:
                mask[ np.where( (pfs.prop('wvl')[w] >= tell[0]) &\
                                (pfs.prop('wvl')[w] <= tell[1]) ) ] = False
            
            #Make sure that there are enough points from which to calculate the continuum
            if pfs.prop('wvl')[w][mask].size < 300:
                sys.stderr.write('Insufficient number of pixels to determine the continuum\n')
                sys.exit()
    
            #Determine the initial B-spline fit, prior to sigma clipping
            #The initial B-spline fit approximates a running mean of the continuum
            #the function considers only interior knots, end knots added automatically
            #returns tuple of knots, B-spline coefficients, and the degree of the spline

            bkpt, tck = self.slatec_splinefit(pfs.prop('wvl')[w][mask], pfs.prop('flux')[w][mask], 
                pfs.prop('ivar')[w][mask], sigma_l, sigma_u, npix, maxiter)
                
            initcont = splev(pfs.prop('wvl'), tck) 
            
            #Check for any negative values of the continuum determination
            #wzero = initcont < 0.
            #if len(initcont[wzero]) > 0: 
            #    initcont[wzero] = np.nan
        
        #Update the PFS object with the initial continuum
        pfs.assign(initcont, 'initcont')
        
        return
        
    def continuum_refinement(self, pfs=None, synth=None, wavelength_range=None, 
        npix=100, sigma_l=3., sigma_u=3., maxiter=5, k=3):

        """
        Refine the continuum normalization by fitting a B-spline to the quotient of the 
        continuum divided, observed spectrum and the best-fit synthethic spectrum. This
        quotient is equivalent to a flat noise spectrum, which should correspond to the
        higher order terms in the continuum. Then divide the CONTINUUM DIVIDED observed spectrum
        by the flat noise spectrum to refine the continuum.
        """
        
        #Regions to continuum normalize
        if wavelength_range is None:
            wavelength_range = [pfs.prop('wvl').min(), pfs.prop('wvl').max()]
            
        #Select the regions within the above specified range
        w = np.where( (pfs.prop('wvl') >= wavelength_range[0]) &\
                      (pfs.prop('wvl') <= wavelength_range[1]) )[0]
        
        m = len(pfs.prop('wvl')[w])
        #Check if the number of data points is larger than the degree of the spline
        if m < k:
            sys.stderr.write(f'Fewer data points (m = {m}) than degree of spline (k = {k})\n')
            sys.exit()
        
        else:
        
            #Define continuum normalized quantities
            fluxi = pfs.prop('flux') / pfs.prop('initcont')
            ivari = pfs.prop('ivar') * pfs.prop('initcont')**2

            #Mask nonsensical values from the continuum determination
            mask = np.full( len(pfs.prop('wvl')), False, dtype=bool)
            
            mask[ np.where( (fluxi > 0.) & (ivari > 0.) &\
                  np.isfinite(ivari) & np.invert(np.isnan(ivari)) &\
                 (synth > 0.) & np.invert(np.isnan(synth)) ) ] = True
            
            bkpt, tck = self.slatec_splinefit(pfs.prop('wvl')[mask], 
                                              fluxi[mask]/synth[mask],
                                              ivari[mask] * synth[mask]**2.,
                                              sigma_l, sigma_u, npix, maxiter) 
            
            noise_spectrum = splev(pfs.prop('wvl'), tck)
            
            refinedcont = pfs.prop('initcont') * noise_spectrum
            
            pfs.assign(refinedcont, 'refinedcont')
            
            return
        
    def check_mask_file_exists(self, filename='', path_blue='specregion/pfs/blue',
        path_red='specregion/pfs/red', mode='lr', root='./'):

        """
        A helper function to check if a spectral mask exists, and if so, to open
        it and return the data.
        """
        #Check if the mask for the blue arm exists
        if not os.path.exists(f'{root}{path_blue}/{filename}.pkl'):
            sys.stderr.write(f'! Spectral mask files not found at {root}{path_blue} \n')
            sys.exit()
        
        #Check if the mask for the red arm exists
        if not os.path.exists(f'{root}{path_red}{mode}/{filename}.pkl'):
            sys.stderr.write(f'! Spectral mask files not found at {root}{path_red}{mode} \n')
            sys.exit()
        
        #Open the mask files for the blue and red arms
        with open(f'{root}{path_blue}/{filename}.pkl', 'rb') as f:
            datab = pickle.load(f)
            
        with open(f'{root}{path_red}{mode}/{filename}.pkl', 'rb') as f:
            datar = pickle.load(f)
        
        #Combine the spectral masks for the blue and red arms of PFS
        data = np.append(datab, datar, axis=0)

        return data
        
    def construct_mask(self, pfs=None, fit_ranges=[[4100., 9100.]]):

        """
        Construct a wavelength mask (in Angstroms) for the given wavelength array, based on 
        the desired wavelength regions to fit.
    
        Parameters
        ----------
        fit_ranges: 2D array-like: wavelength range in Angstroms over which to fit the
                    spectrum for the abundances
              
        Returns
        -------
        mask: array-like: boolean mask of wavelength ranges in Angstroms, where masked regions
              are False
        """

        mask = np.full(len(pfs.prop('wvl')), False, dtype=bool) #initialize boolean mask
    
        #Enforce the specified fit ranges
        for ranges in fit_ranges:
            mask[np.where(((pfs.prop('wvl') >= ranges[0]) & (pfs.prop('wvl') <= ranges[1])))] = True
        
        #Mask regions with nonsensical flux values
        mask[np.where((pfs.prop('flux') < 0.)|(pfs.prop('ivar') <= 0.)|np.isnan(pfs.prop('ivar'))|\
              np.isnan(pfs.prop('flux')))] = False
    
        return mask
        
io = PFSUtilities()