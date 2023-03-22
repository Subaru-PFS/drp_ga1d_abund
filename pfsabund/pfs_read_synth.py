"""
@author: I. Escala (Caltech, Princeton)

Class to read synthetic spectra of Kirby et al. 2008 and Escala et al. 2019a.
"""

from __future__ import absolute_import
import numpy as np
import functools
import gzip
import sys
from pfsabund import pfs_utilities as ut

class ReadSynth():

    def read_synth(self, Teff = np.nan, Logg = np.nan, Feh = np.nan, Alphafe = np.nan, 
        fullres=True, data_path='./', start=4100., sstop=6300.):
    
        """
        Read the ".bin.gz" file containing the synthetic spectrum information
        generated using EMOOG / MOOGIE
    
        Parameters:
        ------------
        Teff: float: effective temperature (K) of synthetic spectrum
        Logg: float: surface gravity (log cm s^(-2)) of synthetic spectrum
        Feh: float: iron abundance ([Fe/H]) (dex) of synthetic spectrum
        Alphafe: float : alpha-to-iron ratio [alpah/Fe] (dex) of synthetic spectrum
        fullres: boolean: if True, then use the unbinned, full-resolution version of the
                          synthetic spectrum
        data_path: string: the path leading to the parent directory containing the synthetic
                           spectrum data
        start: float: start wavelength of synthetic spectrum
        sstop: float: stop wavelength of synthetic spectrum
        file_ext: string: file extension of the filename to be read, default '.bin'
    
        Returns:
        ---------
        wvl: array: the wavelength range covered by the synthetic spectrum, depending on
                    whether it is full resolution or binned, and on stop and start wavelengths
        relflux: array: the normalized flux of the synthetic spectrum
        """
    
        linestart = 3
    
        #Determine which directory to point to (binned/full resolution)
        if fullres == True:
            directory = 'synths/'
            step = 0.02
        else:
            directory = 'bin/'
            step = 0.14


            
        #Check if the parameters are specified, if the filename is not
        if np.all(np.isnan([Teff,Logg,Feh,Alphafe])) == True:
            sys.stderr.write("Error: must define teff, logg, feh, and alphafe")
            sys.exit()
        
        path = data_path+directory #full path name

        #Redefine the parameters according to the file naming convention
        filename = self.construct_title_filename(Teff, Logg, Feh, Alphafe)
        filename = path+filename

        #Open and read the contents of the gzipped binary file without directly
        #unzipping, for enhanced performance
        with gzip.open(filename, 'rb') as f:
            bstring = f.read()
            flux = np.fromstring(bstring, dtype=np.float32)
    
        wvl_range = np.arange(start, sstop+step, step)
        wvl = 0.5*(wvl_range[1:] + wvl_range[:-1])
    
        relflux = 1. - flux
    
        return wvl, relflux
        
    def construct_title_filename(self, Teff=np.nan, Logg=np.nan, Feh=np.nan, Alphafe=np.nan,
        file_ext='.bin', interp=False, Dlam=np.nan):
        
        """
        Helper function to read_synth. Construct the filename corresponding to a given
        set of stellar parameters and elemental abundances
        """

        #Redefine the parameters according to the file naming convention
        if not interp:
            teff = round(Teff/100.)*100
            logg = round(Logg*10.)
            feh = round(Feh*10.)
            alphafe = round(Alphafe*10.)
        else:
            teff = np.round(Teff, decimals=0)
            logg = np.round(Logg*10., decimals=2)
            feh = np.round(Feh*10., decimals=2)
            alphafe = np.round(Alphafe*10., decimals=2)
            dlam = np.round(Dlam*10.,decimals=2)
    
        if logg >= 0.:
            gsign = '_'
        else: gsign = '-'

        if feh >= 0.:
            fsign = '_'
        else: fsign = '-'

        if alphafe >= 0.:
            asign = '_'
        else: asign = '-'
    
        if file_ext == '.bin':
    
            bin_gz_file = "t%2i/g%s%2i/t%2ig%s%2if%s%2ia%s%2i.bin.gz"%(teff,gsign,abs(logg),teff,gsign,abs(logg),fsign,abs(feh),asign,abs(alphafe))
            bin_gz_file = bin_gz_file.replace(" ","0")
            filename = bin_gz_file
        else:
            out_file = "t%2ig%s%2if%s%2ia%s%2i.out2"%(teff,gsign,abs(logg),fsign,abs(feh),asign,abs(alphafe))
            out_file = out_file.replace(" ","0")
            filename = out_file
    
        return filename
        
    def read_interp_synth(self, wvl=np.nan, teff=np.nan, logg=np.nan, feh=np.nan, alphafe=np.nan, 
        fullres=False, data_path='./', start=4100., sstop=6300., hash=None, dlam=np.nan, npar=4):

        """
        Construct a synthetic spectrum in between grid points based on linear interpolation
        of synthetic spectra in the MOOGIE grid
    
        Parameters:
        -----------
        Teff: float: effective temperature (K) of synthetic spectrum
        Logg: float: surface gravity (log cm s^(-2)) of synthetic spectrum
        Feh: float: iron abundance ([Fe/H]) (dex) of synthetic spectrum
        Alphafe: float : alpha-to-iron ratio [alpah/Fe] (dex) of synthetic spectrum
        fullres: boolean: if True, then use the unbinned, full-resolution version of the
                          synthetic spectrum
        data_path: string: the path leading to the parent directory containing the synthetic
                           spectrum data
        start: float: start wavelength of synthetic spectrum
        sstop: float: stop wavelength of synthetic spectrum
        file_ext: string: file extension of the filename to be read, default '.bin'
        hash: dict, optional: a dictionary to use to store memory concerning which synthetic
              spectra have been read in. Should be initliazed externally as an empty dict.
    
        Returns:
        --------
        wvl: array: the wavelength range covered by the synthetic spectrum, depending on
                    whether it is full resolution or binned, and on stop and start wavelengths
        relflux: array: the normalized flux of the synthetic spectrum
        """
    
        #Define the points of the 4D grid


        teff_arr = np.append(np.arange(3500., 5600., 100.), np.arange(5600., 8200., 200.))
        teff_arr = np.round(np.array(teff_arr), decimals=0)
    
        logg_arr = np.round(np.arange(0., 5.5, 0.5), decimals=1)
        feh_arr = np.round(np.arange(-5., 0.1, 0.1), decimals=2)
        alphafe_arr = np.round(np.arange(-0.8, 1.3, 0.1), decimals=2)
        alphafe_arr[8] = 0.

    
        #First check that given synthetic spectrum parameters are in range
        in_grid = self.enforce_grid_check(teff, logg, feh, alphafe)
        if not in_grid: return
               
        params = np.array([teff, logg, feh, alphafe])
        params_grid = np.array([teff_arr, logg_arr, feh_arr, alphafe_arr])
    
        #Now identify the nearest grid points to the specified parameter values
        ds = []; nspecs = []; iparams = []
        for i in range(npar):
    
            #The specified parameter value is a grid point
            w = np.digitize(params[i], params_grid[i])
            if params[i] in params_grid[i]:
                iparam = np.array([w-1, w-1])
                d = [1.]
                nspec = 1
                ds.append(d)
            
            #The specified parameter value is in between grid points
            else:
                if w == (len(params_grid[i])): w -= 1
                iparam = np.array([w-1, w])
                d = params_grid[i][iparam] - params[i]
                d_rev = np.abs(d[::-1])
                nspec = 2
                ds.append(d_rev)
            
            nspecs.append(nspec)
            iparams.append(iparam)
        
        #Now, based on the nearest grid points, construct the linearly interpolated
        #synthetic spectrum
    
        #Determine the number of pixels in a synthetic spectrum based on whether
        #the spectrum is binned or unbinned
        if fullres: 
            step = 0.02
        else: 
            step = 0.14
    
        #ENK: ensure that wavelength array goes to the last pixel, even accounting for floating point rounding error
        wave = np.arange(start, sstop+step, step)
        wave = wave[wave <= sstop+0.01]

        #Calculate the number of pixels in the synthetic spectrum, and initialize the
        #interpolated synthetic spectrum array  --  ENK: final spectrum has length of wvl
        npixels_smooth = len(wvl)
        synth_interp = np.zeros(npixels_smooth)
    
        #Function for loading a specified synthetic spectrum
        def load_synth(p):
    
            teffi, loggi, fehi, alphafei = p
        
            if hash is not None:
        
                #First construct the filename corresponding to the parameters, to use for
                #testing whether we should read in the specified synthetic spectrum
        
                filename = self.construct_title_filename(Teff=teff_arr[teffi], 
                Logg=logg_arr[loggi], Feh=feh_arr[fehi], Alphafe=alphafe_arr[alphafei])
                key = filename[11:-7]
            
                if key not in hash.keys():
                
                    _,synthi = self.read_synth(Teff=teff_arr[teffi], Logg=logg_arr[loggi],
                    Feh=feh_arr[fehi], Alphafe=alphafe_arr[alphafei], fullres=fullres, 
                    data_path=data_path, start=start, sstop=sstop)
                    
                    #ENK: smooth and rebin the spectrum
                    synthi_smooth = ut.io.smooth_gauss_wrapper(wave, synthi, wvl, dlam)
                    
                    hash[key] = synthi_smooth
                
                #If the key is already present in the hash table, then find it and load the data
                else: 
                    synthi_smooth = hash[key]
            
            else:
                _,synthi = self.read_synth(Teff=teff_arr[teffi], Logg=logg_arr[loggi],
                Feh=feh_arr[fehi], Alphafe=alphafe_arr[alphafei], fullres=fullres, 
                data_path=data_path, start=start, sstop=sstop)
                
                #ENK: smooth and rebin the spectrum
                synthi_smooth = ut.io.smooth_gauss_wrapper(wave, synthi, wvl, dlam)
        
            return synthi_smooth
    
        #Load each nearby synthetic spectrum on the grid to linearly interpolate
        #to calculate the interpolated synthetic spectrum
        for i in range(nspecs[0]):
            for j in range(nspecs[1]):
                for k in range(nspecs[2]):
                    for l in range(nspecs[3]):
                
                        p = [iparams[0][i], iparams[1][j], iparams[2][k], iparams[3][l]]
                        synthi = load_synth(p)
                    
                        for m in range(npixels_smooth):
                            synth_interp[m] += ds[0][i]*ds[1][j]*ds[2][k]*ds[3][l]*synthi[m]
                    
        facts = []
        for i in range(npar):
            if nspecs[i] > 1:
                fact = params_grid[i][iparams[i][1]] - params_grid[i][iparams[i][0]]
            else: 
                fact = 1
            facts.append(fact)
        
        synth_interp /= functools.reduce(lambda x, y: x*y, facts)
    
        return wvl, synth_interp
        
    def enforce_grid_check(self, teff=np.nan, logg=np.nan, feh=np.nan, alphafe=np.nan):

        """
        Enforce the grid constraints from Kirby et al. 2009 on a given combination
        of atmospheric model parameters.
    
        Parameters
        ----------
        teff: float: effective temperature
        logg: float: surface gravity
        feh: float: [Fe/H]
        alphafe: float: [alpha/Fe]
    
        Returns
        -------
        in_grid: boolean: if True, the specified parameters are within the K08 grid range
        """
    
        #Check that the effective temperature is within limits
        teff_lims = [3500., 8000.]
        if (teff < teff_lims[0]) or (teff > teff_lims[1]):
            in_grid = False
            return in_grid
        
        logg_hi = 5.0  
        if teff < 7000.: logg_lo = 0.0
        else: logg_lo = 0.5
        logg_lims = [logg_lo, logg_hi]
        
        #Check if the specified surface gravity is within limits
        if (logg < logg_lims[0]) or (logg > logg_lims[1]):
            in_grid = False
            return in_grid
    
        #Check that the specified metallicity is within limits
        feh_lims = [-5., 0.]
    
        #Put checks in place based on the limits of the grid imposed by
        #difficulty in model atmosphere convergence
        teff_vals = np.array([3600., 3700., 3800., 3900., 4000., 4100.])
        feh_thresh = np.array([-4.9, -4.8, -4.8, -4.7, -4.4, -4.6])
        logg_thresh = np.array([[1.5], [2.5, 3.5, 4.0, 4.5], [4.0, 4.5], 
                               [2.5, 3.0, 3.5, 4.0, 4.5, 5.0], [4.5, 5.0], [4.5, 5.0]])
    
        if teff in teff_vals:                       
            where_teff = np.where(teff_vals == teff)[0][0]
            if logg in logg_thresh[where_teff]:
                if (feh < feh_thresh[where_teff]) or (feh > feh_lims[1]):
                    in_grid = False
                    return in_grid
        else:
            if (feh < feh_lims[0]) or (feh > feh_lims[1]):
                in_grid = False
                return in_grid
    
        #Check that the alpha enhancement is within limits  
        alpha_lims = [-0.8, 1.2]
        if (alphafe < alpha_lims[0]) or (alphafe > alpha_lims[1]):
            in_grid = False
            return in_grid
        
        in_grid = True
        return in_grid

io = ReadSynth()
