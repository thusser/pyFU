#!/usr/bin/env python3

# pyfu/flat.py 

"""
Method for calculating the correction for the relative transmissions of the fibres using extracted sky flats.

The OBJTYP term "flatness" was created to distinguish this correction from that of a true flatfield.
"""

import numpy as np
import logging

from astropy.io    import fits
from astropy.table import Table, Column
from astropy.wcs   import WCS

from pyFU.defaults import pyFU_default_formats, pyFU_default_keywords, pyFU_logging_level, pyFU_logging_format
from pyFU.ifu      import Fibre, get_fibres
from pyFU.utils    import hist_integral, merge_dictionaries, get_infiles_and_outfiles, read_tables

logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)

def main () :
	import matplotlib.pyplot as plt
	import sys
	import yaml

	from pyFU.utils        import parse_arguments, read_tables

	logging.info ('*************************** flat ******************************')

	# ---- GET DEFAULT CONFIGURATION AND COMMAND LINE PARAMETERS
	README = """
Script for calculating the correction for the different transmissions of IFU fibres.
	"""
	arguments = {
		'errcol':{'path':'flat:','default':'err_flux',
				'flg':'-E','type':str,'help':'name of flux error column'},
		'flxcol':{'path':'flat:','default':'wavelength',
				'flg':'-F','type':str,'help':'name of flux column'},
		'infiles':{'path':'flat:','default':None, \
				'flg':'-i','type':str,'help':'input FITS table file(s)'},
		'outfiles':{'path':'flat:','default':None, \
				'flg':'-o','type':str,'help':'output YAML file(s) containing the corrections'},
		'pixcol':{'path':'flat:','default':'pixel',
				'flg':'-P','type':str,'help':'name of pixel column'},
		'pixels':{'path':'flat:','default':None, \
				'flg':'-x','type':list,'help':'integration pixels of output image'},
		'plot':{'path':None,'default':False, \
				'flg':'-p','type':bool,'help':'display resulting image'},
		'scale':{'path':None,'default':False, \
				'flg':'-s','type':bool,'help':'scale corrections by fibre area (show intensity, not flux)'},
		'waves':{'path':'flat:','default':None, \
				'flg':'-w','type':list,'help':'integration wavelengths of output image'},
		'yaml':{'path':None,'default':None, \
				'flg':'-y','type':str,'help':'name of pyFU configuration file'}
		'wavcol':{'path':'flat:','default':'wavelength',
				'flg':'-W','type':str,'help':'name of wavelength column'},
		}
	args,cfg = parse_arguments (arguments)
	info = cfg['flat']

	# ---- GET THE INPUT AND OUTPUT FILES
	infiles,outfiles = get_infiles_and_outfiles (args.infiles,args.outfiles)

	# ---- FOR ALL INPUT AND OUTPUT FILES
	for infile,outfile in zip(infiles,outfiles) :
		logging.info (f'Reading {infile} ...')
		spectra,pheader = read_tables (pathname=infile)
		nspectra = len(spectra)
		med = np.zeros(nspectra)
		pheader['OBJECT'] = 'ifu-flattness'
		pheader['OBJTYP'] = 'flatness'

		# FOR ALL SPECTRA IN A FILE
		for i in range(nspectra) :
			spectrum = spectra[i]
			p = spectrum[info['pixcol']
			w = spectrum[info['wavcol']
			f = spectrum[infi['flxcol']

			# GET MEDIAN FLUX USING WAVELENGTH OR PIXEL REGION
			if 'waves' in info and info['waves'] is not None :
				wav = info['waves']
				pix = [max(int(np.interp(wav[0],w,p)),  0),
				       min(int(np.interp(wav[1],w,p))+1,nspectra)]
			elif 'pixels' in info and info['pixels'] is not None :
				pix = info['pixels']
			else :
				pix = [0,nspectra]
			mask = (p >= pix[0])*(p <= pix[1])
			med[i] = np.nanmedian(f[mask],axis=0)
			spectrum.meta['OBJECT'] = 'ifu-flatness'
			spectrum.meta['OBJTYP'] = 'flatness'

		# GET QE'S AND PUT THEM BACK INTO THE SPECTRA
		qe = med/np.nanmean(med)
		for i in range(nspectra) :
			spectrum = spectra[i]
			f = spectrum[info['flxcol']
			err = spectrum[info['errcol']
			rerr = err/f
			spectrum[info['flxcol']] = qe+f-f
			spectrum[info['errcol']] = rerr*qe

		# OUTPUT FLATFIELD SPECTRA
		write_spectra (outfile, spectra, pheader)

		logging.info ('******************************************************************\n')

if __name__ == '__main__' :
	main ()

