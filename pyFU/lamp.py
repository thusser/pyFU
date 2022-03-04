#!/usr/bin/env python3

# pyfu/lamp.py

import bisect
import logging
import numpy as np
import yaml

from astropy.io    import fits
from astropy.table import Table
from scipy         import signal, optimize
from matplotlib    import pyplot as plt

from numpy.polynomial.polynomial import polyfit,polyval
from numpy.polynomial.legendre   import legfit,legval
from numpy.polynomial.laguerre   import lagfit,lagval
from numpy.polynomial.hermite    import hermfit,hermval

from pyFU.utils    import centroid, read_tables, vectorize, write_tables, Gaussian1D, get_infiles_and_outfiles
from pyFU.display  import show_with_menu
from pyFU.wavcal   import pixel2wave, wave2pixel, dispersion_fit, transfer_wavelengths_by_index

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)

def calibrate_spectra (spectra, wavetable=None,
			pixcol='pixel',wavcol='wavelength',flxcol='flux', model='linear',
			prompt=False, show=False) -> bool :
	"""
	Calibrates a list of spectra in astropy.table.Tables by fitting lines in each spectrum.
	"""
	showed = show
	Rs = []
	R3s = []

	# FOR EACH SPECTRUM...
	for idx in range(len(spectra)) :
		logging.info ('---- analysis of extracted spectrum #{0} ----'.format(idx+1))
		spectrum = spectra[idx]
		hdr  = spectrum.meta
		pix  = spectrum[pixcol]
		flux = spectrum[flxcol]

		# WAVELENGTH CALIBRATION
		waves,xes,errs,widths = fit_lamp_spectrum (pix,flux,wavetable)
		if show :
			fig = plt.figure ()
			plt.style.use ('ggplot')
			plt.tight_layout ()
			plt.xlabel ('pixel')
			plt.ylabel ('Gaussian sigma [pixels]')
			plt.plot (xes,widths,'o')
			plt.title ('Width of lamp lines')
			plt.show ()

		pcoef,cov,redchi2,rms = calibrate_lamp_spectrum (waves,xes,errs, model=model, show=show, reversed=True)
		for i in range(len(pcoef)) :
			hdr['WV2PXC{0:02d}'.format(i)] = (pcoef[i],'{0}-th coefficient of wave(pixel) function'.format(i))
		hdr['WV2PXFUN'] = (model,'wave(pixel) function')
		hdr['WV2PXRC2'] = (redchi2,'reduced chi-square of wave(pixel) calibration')

		pcoef,cov,redchi2,rms = calibrate_lamp_spectrum (waves,xes,errs, model=model, show=show)
		hdr = spectrum.meta
		for i in range(len(pcoef)) :
			hdr['PX2WVC{0:02d}'.format(i)] = (pcoef[i],'{0}-th coefficient of pixel(wave) function'.format(i))
		hdr['PX2WVFUN']  = (model,'pixel(wave) function')
		hdr['PX2WVRC2'] = (redchi2,'reduced chi-square of pixel(wave) calibration')

		# GET MEAN DISPERSION IN nm/pixel
		w1 = pixel2wave (pix,pcoef,model=model)
		w2 = pixel2wave (pix+1,pcoef,model=model)
		disp = np.mean(np.abs(w2-w1))
		logging.info ('mean dispersion is {0:.4f} nm/pixel'.format(disp))
		fwhm  = 2.*np.median(widths*disp)*np.sqrt(2.*np.log(2.))
		logging.info ('median FWHM is {0:.4f} nm'.format(fwhm))
		R = np.median(waves/fwhm)
		logging.info ('median spectral resolution R_FWHM is {0:.2f})'.format(R))
		resol = np.median(waves/(3.*disp))
		logging.info ('median 3-pixel resolution R_3pix is {0:.2f}'.format(resol))

		Rs.append (R)
		R3s.append (resol)

		# APPLY TO SPECTRUM
		w = pixel2wave (pix,pcoef, model=model)
		spectrum[wavcol] = w

		# DISPLAY RESULTS
		if show :
			fig = plt.figure ()
			plt.style.use ('ggplot')
			plt.tight_layout ()
			plt.xlabel ('wavelength [nm]')
			plt.ylabel ('flux')
			plt.plot (w,flux,'-',color='black',label='#{0}'.format(idx+1))
			plt.legend (fontsize='x-small')	# bbox_to_anchor=(1.02,1), loc='upper left', ncol=1, fontsize='x-small')
			if prompt :
				plt.show ()
				ans = input ('(aBORT,sILENT) :').lower()
				if ans.startswith('a') : return False
				elif ans.startswith('s') : show = False
			else :
				reslt = show_with_menu (fig,['no more plots','ABORT'])
				if reslt == 'no more plots' :
					show = False
				elif reslt == 'ABORT' :
					return False
		logging.info ('---- end of analysis of extracted spectrum #{0} ----\n'.format(idx+1))

	mR  = np.median(Rs)
	mR3 = np.median(R3s)
	logging.info ('Median spectral resolution of all spectra : R_3pix={0:.2f}, R_FWHM={1:.2f}'.format(mR3,mR))
	if showed :
		n = len(Rs)
		fig = plt.figure ()
		plt.style.use ('ggplot')
		plt.tight_layout ()
		plt.xlabel ('spectrum index')
		plt.title ('spectral resolutions')
		plt.plot (range(n),Rs,'o',color='black',label='R_FWHM')
		plt.plot (range(n),R3s,'o',color='blue',label='R_3pix')
		plt.plot ([0,n-1],[mR,mR],'--',color='black',label='median R_FWHM={0:.1f}'.format(mR))
		plt.plot ([0,n-1],[mR3,mR3],'--',color='blue',label='median R_3pix={0:.1f}'.format(mR3))
		plt.legend (bbox_to_anchor=(1.02,1), loc='upper left', ncol=1, fontsize='x-small')
		plt.show ()
	return True

def calibrate_lamp_spectrum (waves,xes,errs, model='linear', reversed=False, show=False) :
	"""
	Given a set of wavelengths, pixel positions and their errors, fits x(wave) (reverse=False) or
	wave(x) (reversed=True).
	Note that dispersion_fit returns coefficients,covariance_matrix,red.chi^2, and R.M.S.
	"""
	return dispersion_fit (xes,errs,waves, model=model, show=show, reversed=reversed)

def fit_lamp_spectrum (x,y,tab) :
	"""
	Fits a Gaussian+background to (x,y) data in chunks given by a table.
	"""
	logging.info ('Fit to each emission line:')
	xs = []
	ws = []
	es = []
	ss = []
	for row in tab :
		wav   = row['wavelength']
		if wav > 0. :
			xavg  = row['xavg']
			xleft = row['xleft']
			dx    = row['dx']

			xx = x[xleft:xleft+dx]
			yy = y[xleft:xleft+dx]
		
			# FIT GAUSSIANS TO EACH LINE VIA Gaussian1D :
			a = yy[0]+yy[-1]
			b = np.max(yy)-a
			c = xavg
			d = 5.	
			p0 = [a,b,c,d]
			p,cov = optimize.curve_fit (Gaussian1D,xx,yy,p0=p0)
			a,b,c,d = p
			redchi2 = np.sum(np.abs(yy-Gaussian1D(xx,*p))/(len(yy)-len(p)))
			err = np.sqrt(cov[2][2])
			logging.debug ('\tFit: a+b*np.exp(-(x-c)**2/d**2)')
			logging.debug ('\ta = {0:.6f} +/- {1:.6f}'.format(a,np.sqrt(cov[0][0])))
			logging.debug ('\tb = {0:.6f} +/- {1:.6f}'.format(b,np.sqrt(cov[1][1])))
			logging.debug ('\tc = {0:.6f} +/- {1:.6f}'.format(c,err))
			logging.debug ('\td = {0:.6f} +/- {1:.6f}'.format(d,np.sqrt(cov[3][3])))

			fwhm = 2.*np.abs(d)*np.sqrt(2.*np.log(2.))
			logging.info  ('\twave,pixel,fwhm = {0:.3f}, {1:.3f}, {2:.3f} : red. chi^2 = {2:.3f}'.format(wav,c,fwhm,redchi2))

			xs.append (c)
			ws.append (wav)
			es.append (err)
			ss.append (np.abs(d))

	return np.array(ws),np.array(xs),np.array(es),np.array(ss)

def main () :
	import sys
	from pyFU.utils import parse_arguments, initialize_logging

    # ---- GET DEFAULTS AND PARSE COMMAND LINE
	README = """
Python script that performs a wavelength-calibration using an external CSV table with
	(wavelength,xavg,xleft,dx,...)
and a set of extracted lamp spectra.
	"""
	arguments = {
		'flxcol':{'path':'lamp:','default':'flux', \
			'flg':'-f','type':str,'help':'name of flux table columns'},
		'generic': {'path':None,'default':None, \
			'flg':'-G','type':str,'help':'YAML file for generic lamp calibration info'},
		'infiles':{'path':'lamp:','default':None, \
			'flg':'-i','type':str,'help':'input FITS table(s) to be wavelength calibrated'},
		'model':{'path':'lamp:','default':'quadratic', \
			'flg':'-m','type':str,'help':'dispersion model (linear|quadratic|cubic|exp|power)'},
		'outfiles':{'path':'lamp:','default':None, \
			'flg':'-o','type':str, 'help':'output wavelength calibrated FITS table(s)'},
		'pause':{'path':'lamp:','default':False, \
			'flg':'-P','type':bool,'help':'pause/prompt after every spectral calibration'},
		'pixcol':{'path':'lamp:','default':'pixel', \
			'flg':'-x','type':str,'help':'name of pixel table columns'},
		'plot':{'path':None,'default':False, \
			'flg':'-p','type':bool,'help':'plot result'},
		'save':{'path':'lamp:','default':None, \
			'flg':'-Y','type':int,'help':'YAML file for saving wavelength parameters'},
		'trace':{'path':'lamp:','default':None, \
			'flg':'-T','type':str,'help':'pathname of YAML file containing trace'},
		'wavcol':{'path':'lamp:','default':'wavelength', \
			'flg':'-W','type':str,'help':'name of output wavelength table column'},
		'wavetable':{'path':'lamp:','default':None, \
			'flg':'-t','type':str,'help':'name of CSV file containing (wavelength,xavg,xleft,dx) entries'},
		'yaml':{'path':None,'default':None, \
			'flg':'-y','type':str,'help':'global YAML configuration file for parameters'}
		}
	args,cfg = parse_arguments (arguments)
	info = cfg['lamp']

	# ---- LOGGING
	initialize_logging (config=cfg)
	logging.info ('*************************** lamp ******************************')

	# ---- OUTPUT GENERIC CONFIGURATION FILE?
	if args.generic is not None :
		logging.info ('Appending generic lamp configuration info to'+args.generic)
		with open (args.generic,'a') as stream :
			yaml.dump ({'lamp':info}, stream)
		sys.exit(0)

	# ---- GET INPUT AND OUTPUT FILES
	infiles,outfiles = get_infiles_and_outfiles (args.infiles,args.outfiles)

	# ---- CHECK FOR WAVELENGTH TABLE
	if 'wavetable' not in info or info['wavetable'] is None :
		logging.critical ('No wavelength table given')
		sys.exit(1)
	logging.info (f'reading CSV wavelength table {info["wavetable"]} ...')
	tab = Table.read (info['wavetable'],format='ascii.csv')
	logging.debug (str(tab))
	if len(tab) == 0 :
		logging.critical (f'No entries in wavetable {info["wavetable"]}')
		sys.exit(1)
	for key in ['wavelength','xavg','dx'] :
		if key not in tab.colnames :
			logging.critical (f'column {key} not in wavelength table {info["wavetable"]}')
			sys.exit(1)
	if not 'xleft' in tab.colnames :
		tab['xleft'] = tab['xavg']-0.5*tab['dx']

	# ---- FOR ALL INPUT FILES
	for infile,outfile in zip(infiles,outfiles) :

		# ---- READ IN EXTRACTED LAMP SPECTRUM
		logging.info (f'reading extracted lamp input spectra in {infile} ...')
		spectra,header = read_tables (pathname=infile)
		if len(spectra) == 0 :
			logging.critical (f'No lamp spectra read from {infile}')

		# ---- CALIBRATE EACH SPECTRUM
		if not calibrate_spectra (spectra, wavetable=tab,
				 	pixcol=info['pixcol'],wavcol=info['wavcol'],flxcol=info['flxcol'],
					model=info['model'], prompt=info['pause'], show=args.plot) :
			logging.error ('ABORTED lamp wavelength calibration!')

		# ---- SAVE RESULTS IN NEW FITS TABLE FILE
		elif outfile is not None :
			logging.info (f'Saving to FITS table file {outfile}...')
			if 'out_format' not in info :
				info['out_format'] = None
			write_tables (spectra,header=header,pathname=outfile,fmt=info['out_format'])

		logging.info ('***************************************************************\n')

if __name__ == '__main__' :
	main ()

