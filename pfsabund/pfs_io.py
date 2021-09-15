"""
@author: I. Escala (Carnegie, Princeton)

Shell of the PFS GA 1D Pipeline.
Generates a PFS object dictionary initialized with inputs and outputs of the abundance
pipeline.

Inputs: pfsObject* FITS files containing spectral information
Outputs: pfsAbund* FITS file containing information from PFS object dictionary.

Usage:
-------

Below is an example of proper usage of the ReadPFSObject class. This requires usage of
the PFSObject class.

froms pfsabund import pfs_io as io #import the source code containing the relevant classes

# Specify the tract, patch, catId, objId, and visits according to the PFS data model
# for the 1D combined spectra (PFSObject*.fits) to read in the FITS file and generate a
# PFS objet dictionary

pfs = io.Read.read_fits(catId, patch, catId, objId, visit)

# Write the PFS object dictionary to a FITS file
pfs.write_to()
"""

from __future__ import absolute_import
from datetime import datetime as date
import astropy.table as table
from astropy.io import fits
from astropy import wcs
import numpy as np
import collections
import hashlib
import os





_version = '0.0.5'

#################################################################
#################################################################

class PFSObject(dict):

    """
    Dictionary class to store information concerning the PFS spectrum and
    derived stellar parameters, such as the abundances.
    """
    
    def __init__(self, catId, tract, patch, objId, visits):
    
        """
        Create and initialize the attributes of the PFS object
        
        Parameters
        ----------
        
        tract: int: in the range (0, 99999), specifying an area of the sky
        patch: string: in form "m,n", specifying a region within a tract
        objId: int: a unique 64-bit object ID for an object.  For example, 
                       the HSC object ID from the database.
        catId: int: a small integer specifying the source of the objId. 
                    Currently only 0: Simulated, 1: HSC, are defined.
        visits: int or array-like of int: an incrementing exposure number, unique at any
                                          site.
        """
    
        pfsVisitHash = self.calculate_pfsVisitHash([str(vv).encode() for vv in sorted(visits)])
        
        file_format = '%03d-%05d-%s-%016x-%03d-0x%016x' % (catId, tract, patch, objId,\
                                                           len(visits) % 1000, pfsVisitHash)
                                                         
        self['fileNameFormat'] = file_format
                
    def calculate_pfsVisitHash(self, *args):
         
        """       
        Calculate pfsVisitHash. Based on the calculate_pfsVisitHash() function in
        pfs/datamodel/utils.py
        """

        m = hashlib.sha1()
        for l in list(args):
            m.update(str(l).encode())
            
        return int(m.hexdigest(), 16) & 0xffffffffffffffff
            
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
        """ 
        
        # Define the filename to specify the saved output
        filename = 'pfsAbund-%s' % self['fileNameFormat']
        savefile = out_dir+'/'+filename+'.fits'
        
        # Check if the file already exists. If not, then proceed with generating. 
        if not os.path.exists(savefile):
        
            keys = self.keys()
            t = table.Table() #construct a table for the data contained in the dictionary
        
            for key in keys:
                if key != 'fileNameFormat':
                    column = [self[key]] #formatting for an equal number of rows (nrow = 1)
                    t.add_column(table.Column(name=key, data=column)) #add data to table
        
            # If it does not exist, create the specified output directory
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            # Save the constructed table as a FITS file under the specified filename
            t.write(savefile, format='fits')
            
        #Add keywords to the header regarding file creation date and version of the
        #abundance pipeline
        hdu = fits.open(savefile, mode='update')    
        hdr = hdu[0].header
        
        d = date.today()
        hdr.set('date', f'{d.year}-{d.month}-{d.day} {d.hour}:{d.minute}:{d.second}')
        hdr.set('version', _version)
        
        hdu.close()
        
#########################################################################
#########################################################################
        
class ReadPFSObject():

    def __init__(self):
    
        """
        Set properties for reading in the PFS FITS files
        """
        
        self.hdu_dict = collections.OrderedDict()
        
        self.hdu_dict['combined_flux'] = 1
        
        self.hdu_dict['fluxtbl'] = 8
        self.hdu_dict['wavelength'] = 0
        self.hdu_dict['flux'] = 1
        self.hdu_dict['ivar'] = 2
        
        self.hdu_dict['sky'] = 4
        self.hdu_dict['obs'] = 7

        # MNI  -- BEGIN --
        
        # -- ORIGINAL --
        #self.hdu_dict['obj_id'] = 1
        #self.hdu_dict['ra'] = 1
        #self.hdu_dict['dec'] = 1
        #self.hdu_dict['phot'] = 2
        # -- ORIGINAL -- 

        
        self.hdu_dict['obj_id'] = 3
        self.hdu_dict['ra'] = 3
        self.hdu_dict['dec'] = 3
        self.hdu_dict['phot'] = 3
    
        
        # MNI -- END --

        
    def read_fits(self, catId, tract, patch, objId, visits, file_dir='.'):
    
        """
        Read in the FITS files for a given PFS spectrum object
        
        Parameters
        -----------
        file_dir: string: directory containing the PFS fit files, defaults to current
                          directory
        tract: int: in the range (0, 99999), specifying an area of the sky
        patch: string: in form "m,n", specifying a region within a tract
        objId: int: a unique 64-bit object ID for an object.  For example, 
                    the HSC object ID from the database.
        catId: int: a small integer specifying the source of the objId. 
                    Currently only 0: Simulated, 1: HSC, are defined.
        visits: str or array-like of str: an incrementing exposure number, unique at any
                                          site.
        
        Returns
        --------
        pfs: dict: catalog of pfs spectral parameters
        """
        
        nm_to_ang = 10. # conversion factor between nm and Angstroms
        
        if not isinstance(visits, list) or not isinstance(visits, np.ndarray):
            visits = [visits]
        if not isinstance(visits, np.ndarray):
            visits = np.array(visits)
        
        # Initialize a dictionary to store in the data from the PFS fits files
        pfs = PFSObject(catId, tract, patch, objId, visits)
        self.initialize_inputs(pfs)
        
        # Specify the FITS file for the combined spectra of multiple visits
        # According to the datamodel, the file_dir will ultimately be 
        # catId/tract/patch/pfsObject-*.fits
        
        file = "%s/pfsObject-%s.fits" % ( file_dir, pfs.prop('fileNameFormat') )

        # Read in the FITS file HDU object
        hdu = fits.open(file)
        
        ## Assign the PFS spectral information to the PFS object dictionary ##
        
        # Combined flux
        combined_flux = hdu[self.hdu_dict['combined_flux']].data
        pfs.assign(combined_flux, 'combined_flux') #units of nJy
        
        # construct the wavelength solution for the combined wavelength by parsing the WCS
        # keywords in the HDU for the combined flux
        w = wcs.WCS(hdu[self.hdu_dict['combined_flux']].header) 
        pixarr = np.linspace(0, len(combined_flux), len(combined_flux) )
        combined_wavelength = w.wcs_pix2world(pixarr, 1)[0]
        
        combined_wavelength *= nm_to_ang #convert from units of nm to Angstroms
        pfs.assign(combined_wavelength, 'combined_wvl')
        
        # Read in the unbinned quantities from HDU #2 and separate flux, wavelength, and error 
        # into each spectral arm
        # Note that this information is necessary for continuum normalization
        

        # MNI -- BEGIN --
        # -- ORIGINAL --

        #wavelength = hdu[self.hdu_dict['fluxtbl']].data.field(self.hdu_dict['wavelength']) #units of nm

        #wavelength *= nm_to_ang #convert from nm to Angstroms
        
        #flux = hdu[self.hdu_dict['fluxtbl']].data.field(self.hdu_dict['flux']) #units of nJy
        #std = hdu[self.hdu_dict['fluxtbl']].data.field(self.hdu_dict['ivar']) #units of nJy

        # -- ORIGINAL --

        wavelength0 = hdu[self.hdu_dict['fluxtbl']].data.field(self.hdu_dict['wavelength']) #units of nm

        wavelength0 *= nm_to_ang #convert from nm to Angstroms
        flux0 = hdu[self.hdu_dict['fluxtbl']].data.field(self.hdu_dict['flux']) #units of nJy
        std0 = hdu[self.hdu_dict['fluxtbl']].data.field(self.hdu_dict['ivar']) #units of nJy


        filt = (np.isnan(flux0) == False) & (np.isnan(std0) == False)

        wavelength = wavelength0[filt]
        flux = flux0[filt]
        std = std0[filt]
        # MNI -- END -- 

        
        ## Is this actually the intensity error? That is, the error in units of nJy?
        ## Comparing to the values in the first element of HDU #3 (the covariance re-sampled
        ## onto a grid), I think it is more likely that the units are nJy^2
        
        ivar = std**(-1.) #convert the intensity error into an inverse variance
        
        ## Note that for the simulations I have done, for some reason the LR mode does not
        ## contain 3 x 4096 data points per column in HDU #2 although each corresponding LR 
        ## pfsArm file contains 4096 pixels
        
        pfs.assign(wavelength, 'wvl')
        pfs.assign(flux, 'flux')  
        pfs.assign(ivar, 'ivar')
        
        sky = hdu[self.hdu_dict['sky']].data # units of nJy, same length as combined flux
        pfs.assign(sky, 'sky')

        #Load in the targeting information contained in the PFS configuration file
        
        pfsDesignId = hdu[self.hdu_dict['obs']].data['pfsDesignId'][0]

        # MNI -- BEGIN --

        # -- ORIGINAL --
        #hdu.close()
       
        #visit0 = visits[-1]
        #config = '%s/pfsConfig-0x%016x-%06d.fits' % (file_dir, pfsDesignId, visit0)
        #hdu_config = fits.open(config)
        
        ##Save information concerning object id and RA DEC position
        #obj_id = hdu_config[self.hdu_dict['obj_id']].data['objId'][0]
        #ra = hdu_config[self.hdu_dict['ra']].data['ra'][0]
        #dec = hdu_config[self.hdu_dict['dec']].data['dec'][0]
        
        ##Save photometric information
        #filters = hdu_config[self.hdu_dict['phot']].data['filterName']
        #mags = hdu_config[self.hdu_dict['phot']].data['fiberMag']
        #hdu_config.close()


        # -- ORIGINAL --
        

        obj_id = (hdu[self.hdu_dict['obj_id']].header)['OBJID']
        ra = (hdu[self.hdu_dict['ra']].header)['RA']
        dec = (hdu[self.hdu_dict['dec']].header)['DEC']
        filters = hdu[self.hdu_dict['phot']].data['filterName']
        mags = hdu[self.hdu_dict['phot']].data['fiberMag']
        
        hdu.close()

        # MNI -- END --

        
        filter_names = np.array(['g', 'r', 'i', 'z', 'y'])
        filter_vals = np.array([mags[filters == filter_name][0] if filter_name in filters\
                       else np.nan for filter_name in filter_names])
        
        #Assign the OBJID, RA, and DEC to PFS object dictionary
        pfs.assign(obj_id, 'obj_id')
        pfs.assign(ra, 'ra')
        pfs.assign(dec, 'dec')


        
        #Assign the photometry to the PFS object dictionary
        pfs.assign(filter_vals, 'mag')
        pfs.assign(filter_names, 'filter')
        
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



        # MNI -- BEGIN --
        pfs.assign(np.nan, 'distance_error')
        
        # MNI -- END -- 



        
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
        initcont = np.zeros(4096*3)
        refinedcont = np.zeros(4096*3)
        
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
