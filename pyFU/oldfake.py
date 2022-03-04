#!/usr/bin/env python3

# pyFU/fake.py

import logging
import numpy as np
import sys
import yaml

from numpy.random  import normal, poisson
from astropy.io    import fits
from astropy.table import Table
from display       import show_hdu
from matplotlib    import pyplot as plt

from pyFU.defaults import pyFU_default_keywords, pyFU_default_formats
from pyFU.ifu      import get_fibres, hexpos

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)


def fake_spectral_image (config=None, keywords=None, formats=None, full_header=False, ampl=None, pattern=None) :
	"""
	Create a fake IFU spectral image as a FITS HDU.

	If given, pattern is the name of a FITS table file or astropy.io.ascii.csv-compatible text
	table containing the spectral pattern with labelled columns 'wavelength' and 'flux'.
	Otherwise, a random absorption (ampl < 0), flatfield (ampl=0), or emission (ampl > 0)
	spectrum is created.

	The default parameters are for a rough simulation of MORISOT if a complete pyFU
	configuration dictionary isn't used.
	"""
	if config is None :
		config = {'formats':{},'fake':{},'trace':{},'extract':{}, 'ifu':{}}

	if 'extract' not in config : config['extract'] = {}
	if 'fake'    not in config : config['fake']    = {}
	if 'ifu'     not in config : config['ifu']     = {}
	if 'trace'   not in config : config['trace']   = {}
	if 'formats' not in config :
		if formats is None :
			config['formats'] = {}
		else :
			config['formats'] = formats

	# ---- DICTIONARY OF FITS HEADER KEYWORDS
	if keywords is None :
		keywords = dict(pyFU_default_keywords)

	# ---- CREATE CONFIGURATION

	# ... FORMAT DEFAULTS
	info = config['formats']
	if not 'trace_format' in info :
		info['trace_format'] = pyFU_default_formats['trace_format']
	for key in pyFU_default_formats :
		if key not in info :
			info[key] = pyFU_default_formats[key]

	# ... MOST IMPORTANT PARAMETER FIRST
	info = config['ifu']
	number_fibres = 39
	if 'number_fibres' in info :
		number_fibres = info['number_fibres']

	# ... TRACE DEFAULTS
	info = config['trace']

	bias = 0.
	number_traces = number_fibres
	sigma = 2.0
	sigmas = {'b1':4.0,'b2':4.0}
	spacing = 6.2
	spacings = {'b1':12.4,'b2':12.4}
	d = {'bias':bias,'number_traces':number_traces,'sigma':sigma,'sigmas':sigmas,'spacing':spacing,'spacings':spacings}
	for key,val in d.items() :
		if key in info :
			val = info[key]
		else :
			info[key] = val

	# ... EXTRACTION DEFAULTS
	info = config['extract']
	ron = 2.222		# e-
	gain = 1.111	# e-/ADU
	d = {'ron':ron,'gain':gain}
	for key,val in d.items() :
		if key in info :
			val = info[key]
		else :
			info[key] = val

	# ... FAKE IMAGE DEFAULTS
	info = config['fake']
	shape=(250,2048)
	signal = 1000.
	skew = [1.,1.]
	d = {'shape':shape,'signal':signal,'skew':skew}
	for key,val in d.items() :
		if key in info :
			val = info[key]
		else :
			info[key] = val

	# ... FIBRE DEFAULTS
	labels = {idx:'f{0}'.format(idx) for idx in range(1,number_fibres+1)}
	labels[1] = 'b1'
	labels[2] = 'b2'
	dinner = 100.
	douter = 110.
	dinners = {'b1':200.,'b2':200}
	douters = {'b1':210.,'b2':210}
	bylabel = {'b1':[-1500.,0.], 'b2':[1500.,0.]}
	pos = hexpos(37,douter)
	positions = {}
	for i in range(37) :
		positions[labels[i+3]] = pos[i]

	info = config['ifu'] 
	if not 'focal_positions' in info :
		info['focal_positions'] = positions

	if not 'slit_labels' in info :
		info['slit_labels'] = labels 

	if not 'inner_diameter' in info and not 'inner_diameters' in info :
		info['inner_diameter'] = dinner
		info['inner_diameters'] = dinners

	if not 'outer_diameter' in info and not 'outer_diameters' in info :
		info['outer_diameter'] = douter
		info['outer_diameters'] = douters

	# GET LIST OF FIBRES FROM THE CONFIGURATION DICTIONARY
	fibres = get_fibres (config, keywords=keywords, formats=formats)

	# SHAPE OF IMAGE
	ny,nx = shape[0],shape[1]
	data = np.zeros ((ny,nx))
	xg,yg = np.meshgrid (range(nx),range(ny))

	# GET MODEL SPECTRUM
	if pattern is None and 'pattern' in config['fake'] : 
		pattern = config['fake']['pattern']
	if ampl is None and pattern is not None :
		if pattern.endswith('.fits') or pattern.endswith('.fit') :
			t = Table.read (pattern, format='fits')
		else :
			t = Table.read (pattern, format='ascii.csv')
		f = t['flux']
		nf = len(f)
		xf = range(nf)
		x = np.linspace(0,nf,nx)
		y = np.interp (range(nx),xf,f)

	# ... OR CREATE FAKE EMISSION/ABSORPTION SPECTRUM
	else :
		x = np.arange(nx,dtype=float)
		tau = x-x
		if ampl is not None and ampl > 0. :
			ntau = 100
		else :
			ntau = 1000
		tauavg = ntau*sigma/nx
		for i in range(ntau) :
			tau0 = np.random.random()/tauavg
			sig = sigma*(1.+tau0*np.random.random())
			pos = np.random.random()*nx
			tau += tau0*np.exp(-(pos-x)**2/sig**2)+0.1*tau0**2/(1.+(pos-x)**2/sig**2)
		if ampl is not None and ampl > 0. :
			y = ampl*tau
		else :
			y = (1.-np.exp(-tau))

	# SMEAR OUT SPECTRUM
	sol = np.tile (y,(ny,1))

	# CALCULATE SPECTRA
	y0 = config['trace']['spacing']
	skew = config['fake']['skew']
	a = skew[0]/nx
	b = skew[1]/nx**2
	for i in range(number_fibres) :
		fibre = fibres[i]
		amp = signal*(1.+0.2*(2*np.random.random()-1))
		iy = int(y0)
		delta = 1.
		y = iy+(a*xg+b*xg**2)*delta
		sp = amp * np.exp(-((y-yg)/fibre.sigma)**2) * np.exp(-((xg-nx/2)/(nx/2))**2)
		data += sp*sol
		y0 += fibre.spacing
	mask = np.where(np.isnan(data))
	numnan = len(mask)
	if numnan > 0 :
		logging.info ('fake data has {0} NaN values!'.format(numnan))

	# CREATE NOISE
	n1  = normal(0.,ron,data.shape)
	rav = data.reshape(-1)
	p   = poisson (lam=rav,size=(1,rav.size))
	n2  = p.reshape (shape)
	mask = np.where(np.isnan(n2))
	numnan = len(mask)
	if numnan > 0 :
		logging.info ('fake noise has {0} NaN values!'.format(numnan))

	# ADD NOISE TO DATA
	data += n1+n2

	# ADD BIAS
	logging.info ('adding bias level {0}'.format(bias))
	data += bias

	# CREATE FITS HDU
	hdu = fits.PrimaryHDU(data)

	# ADD METADATA
	if full_header :
		hdr = hdu.header
		if 'number_fibres' in keywords :
			hdr[keywords['number_fibres'][0]] = number_fibres,'number of spectra'
		if 'ron' in keywords :
			hdr[keywords['ron'][0]] = ron,'read-out-noise [e-]'
		if 'gain' in keywords :
			hdr[keywords['gain'][0]] = gain,'gain [e-/ADU]'
		# ADD FAKE SKEW METADATA
		hdr['IFU-SKW0'] = skew[0],'linear slope factor'
		hdr['IFU-SKW1'] = skew[1],'quadratic slope factor'
		for fibre in fibres  :
			fibre.update_header (hdr,keywords)

	# RETURN HDU
	return hdu


def main () :
	from pyFU.utils import parse_arguments, initialize_logging

	# ---- GET CONFIGURATION AND COMMAND LINE PARAMETERS
	README = """
Script to create fake IFU spectral images.
	"""
	arguments = {
		'full':    {'path':None,'default':None, \
					'flg':'-H','type':bool,'help':'include full pyFU image header'},
		'generic': {'path':None,'default':None, \
					'flg':'-G','type':str,'help':'YAML file for generic fake image configuration info'},
		'outfile': {'path':'fake:','default':None, \
					'flg':'-o','type':str,'help':'path of (optional) output FITS file'},
		'pattern': {'path':'fake:','default':None, \
					'flg':'-P','type':str,'help':'pattern to use: solar|absorption|emission|flat'},
		'plot':    {'path':None,'default':False, \
					'flg':'-p','type':bool,'help':'plot result'},
		'yaml':    {'path':None,'default':None, \
					'flg':'-y','type':str,'help':'global YAML configuration file for parameters'}
		}
	args,cfg = parse_arguments (arguments, readme=README)
	info = cfg['fake']

	# ---- LOGGING
	initialize_logging (config=cfg)
	logging.info ('******************************* fake **************************')
	logging.debug (f'info:{info}')

	# ---- OUTPUT GENERIC CONFIGURATION FILE?
	if args.generic is not None :
		logging.info ('Appending generic fake image configuration info to '+args.generic)
		with open (args.generic,'a') as stream :
			yaml.dump ({'fake':info}, stream)
		sys.exit(0)

	# ---- CREATE HDU
	if args.pattern == 'solar' :			# REAL SOLAR SPECTRUM
		hdu = fake_spectral_image (config=cfg, full_header=args.full)
	elif args.pattern == 'emission' :		# RANDOM EMISSION SPECTRUM
		ampl = 100.
		hdu = fake_spectral_image (config=cfg, full_header=args.full, ampl=ampl)
	elif args.pattern == 'absorption' :		# RANDOM ABSORPTION SPECTRUM
		ampl = -1.
		hdu = fake_spectral_image (config=cfg, full_header=args.full, ampl=ampl)
	else :									# USER'S SPECTRUM
		hdu = fake_spectral_image (config=cfg, full_header=args.full, pattern=args.pattern)

	# ---- SHOW
	if args.plot :
		show_hdu (hdu, aspect='auto',colourbar=True)
		if 'outfile' in info and info['outfile'] is not None :
			plt.title (info['outfile'])
		plt.show()

	# ---- SAVE TO FILE
	if 'outfile' in info and info['outfile'] is not None :
		hdu.header['FILENAME'] = info['outfile'],'pathname of fake IFU spectral image'
		logging.info ('Saving to {0} ...'.format(info['outfile']))
		hdus = fits.HDUList([hdu])
		hdus.writeto (info['outfile'],overwrite=True)


if __name__ == '__main__' :
	main ()

