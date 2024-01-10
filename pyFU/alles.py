#!/usr/bin/env python3

# pyfu/alles.py

import argparse
import logging
import matplotlib as mpl
import numpy as np
import sys
import yaml

from astropy.io   import fits
from matplotlib   import pyplot as plt

from pyFU.display import show_hdu
from pyFU.extract import SpectrumExtractor
from pyFU.trace   import SpectrumTracer
from pyFU.utils   import merge_dictionaries, get_infiles_and_outfiles
from pyFU.utils   import write_tables, initialize_logging

def main (infiles, outfiles, cfg_file, tracecfg_file) :

	# ---- LOGGING
	initialize_logging ()
	s = 10*'*'+' extract and image '+10*'*'
	logging.info (s)

	# ---- GET IFU CONFIGURATION
	try :
		with open (cfg_file) as stream :
			cfg = yaml.safe_load (stream)
	except :
		logging.error ('cannot read configuration YAML file!')
		sys.exit(1)

	# ---- GET TRACE CONFIGURATION
	try :
		with open (tracecfg_file) as stream :
			tracecfg = yaml.safe_load (stream)
	except :
		logging.error ('cannot read trace YAML file!')
		sys.exit(1)
	merge_dictionaries (cfg,tracecfg)

	if 'extract' in cfg :
		extract_cfg = cfg['extract']
	else :
		extract_cfg = cfg

	# ---- GET INPUT AND OUTPUT FILE NAMES
	infiles,outfiles = get_infiles_and_outfiles (infiles,outfiles,cfg=info)

	# ---- FOR ALL INPUT FILES
	for infile,outfile in zip(infiles,outfiles) :
		logging.info (f'Extracting {outfile} from {infile}')

		# ---- READ IN DATA 
		hdus = fits.open (infile)
		hdu = hdus[0]

		# ---- CREATE TRACER
		tracer = SpectrumTracer (hdu,config=cfg)

		# ---- EXTRACT SPECTRA
		spectra = extractor.extract (hdu)
		nspectra = len(spectra)
		if nspectra == 0 :
			logging.error ('No spectra extracted!')
			sys.exit(1)

		# ---- GET STRIPPED HEADER
		logging.info ('Transfering header ...')
		hdr = {}
		extractor.filter.copy_header (hdu.header,hdr,bare=True,comments=False)

		# ---- WRITE TO FITS BINARY TABLE FILE
		logging.info ('Saving to FITS table file {0} ...'.format(outfile))
		write_tables (spectra,outfile, header=hdr,overwrite=True, keywords=extractor.keywords) 

		# ---- CONVERT SPECTRA TO A matplotlib GRAPH AND HENCE TO A PIXEL IMAGE
		info = cfg['image']
		title = f'{infile} (mean)'
		logging.info (f'Producing mean focal-plane image {outfile}')
		spectra2image (spectra, title=title, outfile=outfile, info=metadata,
				show=args.plot, logscale=info['logscale'], fudge=info['fudge'],
				scale=args.scale, width=args.width,
				labeled=info['labeled'])

		logging.info (len(s)*'*')

if __name__ == '__main__' :
	main ()
