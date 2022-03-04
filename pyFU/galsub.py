#!/usr/bin/env python3

# pyFU/galsub.py

"""
Subtracts a sky and galaxy background from an flat-fielded IFU spectral image by optimizing
the relative positions of a background-subtracted galaxy image and the IFU fluxes in the band-pass
used by the image.  In this first version, the intrinsic PSF of the IFU and reference images are
assumed to be identical.
"""

import sys
import logging
import numpy as np
import yaml

from astropy.table import Table,Column
from astropy       import units  as u
from matplotlib    import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
from pyFU.display import show_with_menu
from pyFU.utils   import read_tables, parse_arguments, write_tables

logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)

# ---- GET PARAMETERS FROM CONFIGURATION FILE OR COMMAND LINE (LATTER HAS HIGHER PRIORITY)
README = """
Python script to subtract a complex background from a set of calibrated IFU spectra using an
image to model the spatial structure.
"""

arguments = {
	'errcol':     {'path':'galsub:','default':'err_flux','dshow':'err_flux','flg':'-E','type':str,'help':'name of flux error column)'},
	'flxcol':     {'path':'galsub:','default':'flux','dshow':'flux','flg':'-F','type':str,'help':'name of flux column)'},
	'infile':     {'path':'galsub:','default':None,'dshow':None,'flg':'-i','type':str,'help':'input FITS binary table containing spectra'},
	'filter':     {'path':'galsub:','default':None,'dshow':None,'flg':'-f','type':str,'help':'pathname of file containing filter function'},
	'outfile':    {'path':'rebin:','default':None,'dshow':None,'flg':'-o','type':str,'help':'pathname of output FITS table'},
	'plot':       {'path':None,'default':False,'dshow':False,'flg':'-p','type':bool,'help':'plot result'},
	'pixcol':     {'path':'rebin:','default':'pixel','dshow':'pixel','flg':'-P','type':str,'help':'name of pixel column)'},
	'tcol':       {'path':'galsub:','default':'transmission','dshow':'transmission','flg':'-T','type':str,'help':'name of filter transmission column)'},
	'reference':  {'path':'rebin:','default':None,'dshow':None,'flg':'-R','type':str,'help':'pathname of reference image'},
	'wavcol':     {'path':'rebin:','default':'wavelength','dshow':'wavelength','flg':'-w','type':str,'help':'name of wavelength column'},
	'wcol':       {'path':'rebin:','default':'wavelength','dshow':'wavelength','flg':'-W','type':str,'help':'name of filter wavelength column'},
	'yaml':       {'path':'rebin:','default':None,'dshow':None,'flg':'-y','type':str,'help':'global YAML configuration file for parameters'}
	}
args,cfg = parse_arguments (arguments, readme=README)
info = cfg['galsub']

logging.info ('*************************** galsub ******************************')

# ---- GET LIST OF INPUT SPECTRA
if not 'infile' in info or info['infile'] is None :
	logging.critical ('No input table file given!')
	sys.exit(1)
logging.info ('Getting the table file {0} ...'.format(info['infile']))
spectra,hdr = read_tables (info['infile'])
if len(spectra) == 0 :
	logging.error ('no spectra in {0}'.format(info['infile']))
	sys.exit(1)

# ---- GET REFERENCE IMAGE
if 'reference' in info and info['reference'] is not None :
	logging.info ('using reference {0}'.format(info['reference']))
	hdu = fits.open (info['reference'])[0]

# ---- GET REFERENCE FILTER
if 'filter' in info and info['filter'] is not None :
	logging.info ('using filter curve in {0}'.format(info['filter']))
	filter = read_spectrum (info['filter'])

# ---- CONVOLVE EACH SPECTRUM WITH THE FILTER FUNCTION

fluxes = []
for i in range(len(spectra)) :
	spectrum = spectra[i]
	hdr = spectrum.meta
	fibre = Fibre (header=hdr, keywords=keywords, formats=formats)
	flx = convolve_table (spectrum, filter, \
			pixcol=info['pixcol'], wavcol=info['wavcol'], flxcol=info['flxcol'], errcol=info['errcol'],
			pcol=info['pcol'], wcol=info['wcol'], fcol=info['tcol'], ecol=None)
	fluxes.append ( {'index':fibre.index, 'label':fibre.label, 'position':fibre.pos, 'flux':flx} )

# ---- FIND THE BEST POSITION OFFSET TO EXPLAIN FILTER FLUXES

# ---- SAVE RESULTS
if info['outfile'] is not None :
	logging.info ('Saving results to {0} ...'.format(info['outfile']))
	write_tables (results,pathname=info['outfile'])

