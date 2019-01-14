"""
@author: I. Escala (Caltech, Princeton)

Shell of the PFS GA 1D Pipeline.
Generates a PFS object dictionary initialized with inputs and outputs of the abundance
pipeline.

Inputs: pfsArm* FITS files containing spectral information and user-specified resolution
mode.
Outputs: pfsAbund* FITS file containing information from PFS object dictionary.

Usage:
-------

Below is an example of proper usage of the ReadPFSObject class. This requires usage of
the PFSObject class.

import pfs_io as io #import the source code containing the relevant classes

#Specify the unique identifier according to the PFS data model (visit) for the PFS 1d 
#spectra (e.g., pfsArm-000001-b1.fits) and the resolution mode of the red arm to read
#in the FITS files and generate PFS object dictionary

pfs = io.Read.read_fits(visit=000001, mode='mr') 

#Write the PFS object dictionary to a FITS file with the filename pfsAbund-000001.fits
pfs.write_to('pfsAbund-000001')
"""

from __future__ import absolute_import
import astropy.table as table
from astropy.io import fits
import numpy as np
import collections
import os

#################################################################
#################################################################

class PFSObject(dict):

    """
    Dictionary class to store information concerning the PFS spectrum and
    derived stellar parameters, such as the abundances.
    """
    
    def __init__(self):
    
        """
        Create and initialize the attributes of the PFS object
        """
        pass
        
    def prop(self, property_name=''):
    
        """
        Retrieve property from self dictionary
        
        Parameters
        ----------
        property_name: string: name of property
        
        Returns
        -------
        values: array: numpy array of property values
        """
        
        if property_name in self:
            values = self[property_name]
            return values
            
    def assign(self, value, property_name=''):
    
        """
        Assign a value to a property of the self dictionary
        
        Parameters
        ----------
        property_name: string: name of property
        value: array-like, float, or string: value to assign to property
        """
    
        self[property_name] = value
        
    def write_to(self, out_dir='.'):
    
        """
        Write the dictionary to a FITS file
        
        Parameters
        ----------
        filename: string: the filename of the output fits file, omitting the file extension
        """ 
        
        #Define the filename to specify the saved output
        filename = 'pfsAbund-%06d' % self['visit']
        savefile = out_dir+'/'+filename+'.fits'
        
        #Check if the file already exists. If not, then proceed with generating. 
        if not os.path.exists(savefile):
        
            keys = self.keys()
            t = table.Table() #construct a table for the data contained in the dictionary
        
            for key in keys:
                column = [self[key]] #formatting for an equal number of rows (nrow = 1)
                t.add_column(table.Column(name=key, data=column)) #add data to table
        
            #If it does not exist, create the specified output directory
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            #Save the constructed table as a FITS file under the specified filename
            t.write(savefile, format='fits')
        
#########################################################################
#########################################################################
        
class ReadPFSObject():

    def __init__(self):
    
        """
        Set properties for reading in the PFS FITS files
        """
        
        self.hdu_dict = collections.OrderedDict()
        self.hdu_dict['flux'] = 1
        self.hdu_dict['ivar'] = 2
        self.hdu_dict['wavelength'] = 4
        self.hdu_dict['sky'] = 5
        self.hdu_dict['config'] = 6
        self.hdu_dict['obj_id'] = 1
        self.hdu_dict['ra'] = 1
        self.hdu_dict['dec'] = 1
        self.hdu_dict['phot'] = 1
            
    def read_fits(self, visit, file_dir='.', mode=''):
    
        """
        Read in the FITS files for a given PFS spectrum object
        
        Parameters
        -----------
        visit: int: an incrementing 6-digit exposure number, unique at any site
        file_dir: string: directory containing the PFS fit files
        mode: string: resolution mode of the PFS red arm
        
        Returns
        --------
        pfs: dict: catalog of pfs spectral parameters
        """
    
        #Define the spectrograph number
        spectrograph = 1
        nm_to_ang = 10. #conversion factor between nm and Angstroms
        
        #Initialize a dictionary to store in the data from the PFS fits files
        pfs = PFSObject()
        
        self.initialize_inputs(pfs)
        pfs.assign(visit, 'visit')
        
        #Specify the red arm resolution mode
        assert mode in ['mr', 'lr', 'MR', 'LR']
        pfs.assign(mode.lower(), 'mode')
        
        #Based on the resolution mode of the red arm, specify the relevant file
        if pfs.prop('mode') == 'mr':
            red_arm = 'm'
        if pfs.prop('mode') == 'lr':
            red_arm = 'r'
        
        #Loop over each of the three arms of PFS
        arms = ['b', red_arm, 'n']
        
        #Initialize lists to store flux, inverse variance, wavelength, and
        #sky-subtracted 1D information for all 3 arms of PFS
        flux = []; ivar = []; wvl = []; sky = []
        
        for arm in arms:
        
            #Using pfsArm files instead of pfsObject files (at least for now) for
            #clarity in separating spectral information for each arm. Additionally,
            #the pfsObject files appear to be re-sampled onto a regular wavelength grid.
        
            #Specify the FITS file for the relevant arm
            #pfsArm: Reduced but not combined single spectra from a single exposure 
            #(flux and wavelength calibrated)
            file = '%s/pfsArm-%06d-%1s%1d.fits' % (file_dir, visit, arm, spectrograph)
        
            #Read in the FITS file HDU object
            hdu = fits.open(file)
    
            #Identify the flux and covariance arrays
            fluxi = hdu[self.hdu_dict['flux']].data[0]  #units of nJy
            covari = hdu[self.hdu_dict['ivar']].data[0][0]
            ivari = np.reciprocal(covari)
            wavei = hdu[self.hdu_dict['wavelength']].data[0]
            wavei *= nm_to_ang #convert to Angstroms
            skyi = hdu[self.hdu_dict['sky']].data[0] #also units of nJy
            
            flux.append(fluxi)
            ivar.append(ivari)
            wvl.append(wavei)
            sky.append(skyi)
            
        #Load in the targeting information contained in the PFS configuration file
        pfsConfigId = hdu[self.hdu_dict['config']].data['pfsConfigId'][0]
        hdu.close()
       
        hexbase = '0x' + '%08x'.zfill(12) % pfsConfigId
        config = '%s/pfsConfig-%s.fits' % (file_dir, hexbase)
        hdu_config = fits.open(config)
        
        #Save information concerning object id and RA DEC position
        obj_id = hdu_config[self.hdu_dict['obj_id']].data['objId'][0]
        ra = hdu_config[self.hdu_dict['ra']].data['ra'][0]
        dec = hdu_config[self.hdu_dict['dec']].data['dec'][0]
        
        #Save photometric information
        g, r, i, z, y = hdu_config[self.hdu_dict['phot']].data['fiberMag'][0]
        hdu_config.close()
        
        #Assign the OBJID, RA, and DEC to PFS object dictionary
        pfs.assign(obj_id, 'obj_id')
        pfs.assign(ra, 'ra')
        pfs.assign(dec, 'dec')
        
        #Assign the photometry to the PFS object dictionary
        pfs.assign(g, 'g')
        pfs.assign(r, 'r')
        pfs.assign(i, 'i')
        pfs.assign(z, 'z')
        pfs.assign(y, 'y')
               
        #Convert the lists to arrays
        flux = np.array(flux); ivar = np.array(ivar); wvl = np.array(wvl); sky = np.array(sky)
            
        #Assign the PFS spectral information to the PFS object dictionary
        pfs.assign(flux, 'flux')
        pfs.assign(ivar, 'ivar')
        pfs.assign(wvl, 'wvl')
        pfs.assign(sky, 'sky')
        
        self.initialize_outputs(pfs)
        
        return pfs
        
    def initialize_inputs(self, pfs):
    
        """
        Initialize the PFS object dictionary with the inputs to the abundance pipeline
        Later should include specific ways to read in velocity pipeline outputs and photometric
        information based on relevant formatting
        """
        
        #Assume that we have the distance to the star
        pfs.assign(np.nan, 'distance') #Distance to the star, based on RR Lyrae, Gaia, photometric parallax, etc.
        
        #Assume that the velocity pipeline exists and we have all the outputs
        p_vhelio = np.zeros( (20,500) )
        template = np.zeros(3*4096) #assume the template is sampled on the same wavelength grid as the full PFS spectrum for now
        
        pfs.assign(np.nan, 'vhelio') #Heliocentric velocity
        pfs.assign(np.nan, 'verr') #Error on the velocity
        pfs.assign(p_vhelio, 'p_vhelio') #Complete posterior probability distribution for vhelio
        pfs.assign(template, 'vtemplate') #Best-fit radial velocity template
        pfs.assign(np.nan, 'vchi') #Chi-squared between the observed spectrum and best fit template
        pfs.assign(1, 'vflag') #Flag for bad value of the chi-squared, boolean value
        pfs.assign(np.nan, 'sn') #Signal-to-noise estimate (e.g., from continuum regions)
        
        return pfs
        
    def initialize_outputs(self, pfs):
        
        """
        Initialize empty arrays and values for the final returned PFS object
        NOTE: These values will be assigned later in the development of the 
        PFS GA 1D pipeline
        """
        
        #Assuming an array of ages of 6 - 13 Gyr in 1 Gyr increments
        nages = 8
        teffphot = np.zeros(nages)
        teffphoterr = np.zeros(nages)
        loggphot = np.zeros(nages)
        loggphoterr = np.zeros(nages)
        vturbphot = np.zeros(nages)
        mhphot = np.zeros(nages)
        
        pfs.assign(teffphot, 'teffphot') #photometric effective temperature for each age
        pfs.assign(teffphoterr, 'teffphoterr') #error on teffphot for each age
        pfs.assign(loggphot, 'loggphot') #photometric surface gravity for each age
        pfs.assign(loggphoterr, 'loggphoterr') #error on loggphot for each age
        pfs.assign(vturbphot, 'vturbphot') #photometric microturbulent velocity for each age
        pfs.assign(mhphot, 'mhphot') #photometric metallicity [M/H] for each age
        
        #Continuum as a function of wavelength, of size (3,4096) (narm, nfiber)
        initcont = np.zeros((3, 4096))
        refinedcont = np.zeros((3, 4096))
        
        pfs.assign(initcont, 'initcont') #initial continuum guess (before abundance measurement)
        pfs.assign(refinedcont, 'refinedcont') #refined (final) continuum, after abundance measurement
        
        #Spectroscopic atmospheric parameters
        pfs.assign(np.nan, 'teff') #spectroscopic effective temperature
        pfs.assign(np.nan, 'logg')  #adopted photometric surface gravity (for well known distances) or otherwise determined final logg
        pfs.assign(np.nan, 'vturb') #adopted microturbulent velocity
        pfs.assign(np.nan, 'mh') #adopted spectroscopic [M/H]
        pfs.assign(np.nan, 'alphafe') #adopted spectroscopic [alpha/Fe]
        
        #Abundances
        pfs.assign(np.nan, 'feh') #[Fe/H]
        pfs.assign(np.nan, 'cfe') #[C/Fe]
        pfs.assign(np.nan, 'mgfe') #[Mg/Fe]
        pfs.assign(np.nan, 'cafe') #[Ca/Fe]
        pfs.assign(np.nan, 'sife') #[Si/Fe]
        pfs.assign(np.nan, 'tife') #[Ti/Fe]
        pfs.assign(np.nan, 'mnfe') #[Mn/Fe]
        pfs.assign(np.nan, 'cofe') #[Co/Fe]
        pfs.assign(np.nan, 'nife') #[Ni/Fe]
        pfs.assign(np.nan, 'bafe') #[Ba/Fe]
        
        #Errors on all spectroscopic/abundance quantities
        pfs.assign(np.nan, 'tefferr') 
        pfs.assign(np.nan, 'loggerr') #if an error is present?
        pfs.assign(np.nan, 'vturberr')
        pfs.assign(np.nan, 'mherr')
        pfs.assign(np.nan, 'alphafeerr')
        pfs.assign(np.nan, 'feherr') 
        pfs.assign(np.nan, 'cfeerr')
        pfs.assign(np.nan, 'mgfeerr')
        pfs.assign(np.nan, 'cafeerr')
        pfs.assign(np.nan, 'sifeerr')
        pfs.assign(np.nan, 'tifeerr')
        pfs.assign(np.nan, 'mnfeerr')
        pfs.assign(np.nan, 'cofeerr')
        pfs.assign(np.nan, 'nifeerr')
        pfs.assign(np.nan, 'bafeerr')
        
        #Unique identifiers from cross-matched catalogs
        pfs.assign('2345', 'gaia_id') #Gaia ID
        pfs.assign('6789', '2mass_id') #2MASS ID
        
        return pfs
            
Read = ReadPFSObject()