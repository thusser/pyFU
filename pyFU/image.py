#!/usr/bin/env python3

# pyfu/image.py 

"""
Method for turning extracted spectra into IFU images.
"""

import numpy as np
import logging

from astropy.io    import fits
from astropy.table import Table, Column
from astropy.wcs   import WCS

from matplotlib              import pyplot as plt, cm as colourmaps
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyFU.defaults import pyFU_default_formats, pyFU_default_keywords
from pyFU.ifu      import Fibre, get_fibres
from pyFU.utils    import hist_integral, merge_dictionaries, get_infiles_and_outfiles

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)


def spectra2image (spectra, title=None, pixels=None, waves=None, outfile=None, show=False, scale=False,
					info=None, logscale=False, cmap='viridis', fudge=1.35,
					keywords=None, formats=None, width=10., labeled=False) :
	"""
	Converts a list of extracted IFU spectra into an IFU focal plane image.
	The fudge factor helps to give the image the desired shape, depending upon the IFU arrangement.
	If the dictionary "info" is given, then this metadata is displayed to the right of the image.
	"""
	nspectra = len(spectra)
	if keywords is None : keywords = pyFU_default_keywords
	if formats  is None : formats  = pyFU_default_formats

	# GET SIZES
	sizes = np.full(nspectra,np.nan,dtype=float)
	msize = None
	for i in range(nspectra) :
		spectrum = spectra[i]
		hdr = spectrum.meta
		fibre = Fibre(header=hdr,keywords=keywords,formats=formats)
		if fibre is None or fibre.inner_diameter is None :
			logging.error (f'unable to instatiate fibre #{i+1}')
		elif np.isnan(fibre.inner_diameter) :
			sizes[i] = np.nan
		else :
			sizes[i] = fibre.inner_diameter
	if not np.all(np.isnan(sizes)) :
		msize = np.nanmedian(sizes)
	logging.info (f'median fibre size: {msize}')
	logging.info (f'min.   fibre size: {np.nanmin(sizes)}')
	logging.info (f'max.   fibre size: {np.nanmax(sizes)}')

	# GET RANGE OF FLUXES
	fluxes = np.zeros(nspectra)
	for idx in range(nspectra) :
		spectrum = spectra[idx]
		f = spectrum['flux']

		area = 1.0
		if scale and (msize is not None) and (not np.isnan(sizes[idx])) :
			area = (sizes[idx]/msize)**2
			f /= area # SCALE TO COMMON AREA
			logging.info (f'scaling #{idx} by {area}')

		if waves is not None :
			w = spectrum['wavelength']
			if waves[0] is None :
				wav1 = w[0]
			else :
				wav1 = waves[0]
			logging.debug (f'minimum wavelength: {wav1:.3f}')
			if waves[1] is None :
				wav2 = w[-1]
			else :
				wav2 = waves[1]
			logging.debug (f'maximum wavelength: {wav2:.3f}')
			val = np.nansum(f[np.where ((w>=wave1)*(w<=wave2))])
			logging.info (f'wavelength integrated flux, rel. area for fibre #{idx}: {val:.3e},{area:.2f}')
		elif pixels is not None :
			p = spectrum['pixels']
			if pixels[0] is None :
				pix1 = pixels[0]
			else :
				pix1 = pixels[0]
			logging.debug ('minimum pixel : {0:.3f}'.format(pix1))
			if pixels[1] is None :
				pix2 = pixels[1]
			else :
				pix2 = pixels[1]
			logging.debug ('maximum pixel : {0:.3f}'.format(pix2))
			val = np.nansum(f[np.where ((p>=pix1)*(p<=pix2))])
			logging.info (f'pixel integrated flux, rel. area for fibre #{idx}: {val:.3e},{area:.2f}')
		else :
			val = np.nanmean(f)
			logging.info (f'mean flux, rel. area for fibre #{idx}: {val:.3e},{area:.2f}')
		fluxes[idx] = val

	fluxes = np.array(fluxes)
	fmin = np.nanmin(fluxes)
	fmax = np.nanmax(fluxes)
	logging.info (f'fmin={fmin:.3e}, fmax={fmax:.3e}')
	if logscale :
		if fmax <= 0. :
			fmax += fmin
			logging.info (f'added {fmin:.3e} to grayscaling values')
			fmin = 0.
		if fmin <= 0. :
			fmin = 0.001*np.abs(fmax-fmin)
			logging.info (f'set grayscaling basement to {fmin:.3e}')

	# GET matplotlib COLORMAP
	cmap = colourmaps.get_cmap (cmap,128)

	# CREATE PLOT
	plt.style.use ('dark_background')
	fig,ax = plt.subplots ()
	ax.set_aspect ('equal')
	xlim = [0.,0.]
	ylim = [0.,0.]
	grays = np.zeros(nspectra)
	posits = []
	dins = []

	xdummy = []
	ydummy = []
	cdummy = []
	for i in range(nspectra) :
		spectrum = spectra[i]
		f = fluxes[i]
		if scale and (msize is not None) and (not np.isnan(sizes[i])) :
			f *= (msize/sizes[i])**2		# SCALE TO COMMON AREA
		if logscale :
			gray = (np.log(f)-np.log(fmin))/(np.log(fmax)-np.log(fmin))
		else :
			gray = (f-fmin)/(fmax-fmin)
		fibre = Fibre (header=spectrum.meta)
		pos = fibre.pos
		din = fibre.inner_diameter
		if pos is not None and din is not None :
			if xlim[0] > pos[0]-din : xlim[0] = pos[0]-din
			if xlim[1] < pos[0]+din : xlim[1] = pos[0]+din
			if ylim[0] > pos[1]-din : ylim[0] = pos[1]-din
			if ylim[1] < pos[1]+din : ylim[1] = pos[1]+din
			c = cmap(gray)
			circle = plt.Circle (pos,din/2,color=c)	# ,edgecolor='white')
			ax.add_artist (circle)
			if labeled :
				plt.text (pos[0],pos[1],fibre.label,horizontalalignment='center',verticalalignment='center',fontsize=8)
			xdummy.append(pos[0])
			ydummy.append(pos[1])
			cdummy.append(cmap(gray))
	plt.xlim (xlim)
	plt.ylim (ylim)

	# SHOW METADATA?
	if info is not None and len(info) > 0 :
		x = xlim[0]+0.01*(xlim[1]-xlim[0])
		dy = 0.1*(ylim[1]-ylim[0])
		y = ylim[1]-dy
		for key in info :
			logging.info (f'{key} : {info[key]} @ {x},{y}')
			plt.text (x,y,f'{key} : {info[key]}',fontsize=8)
			y -= dy
 
	# SET THE FIGURE PROPORTIONS
	ratio = fudge*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])
	logging.debug (f'figure ratio: {ratio:.2f}')
	if ratio < 1. :
		fig.set_figwidth (width)
		fig.set_figheight (width*ratio)
	else :
		fig.set_figwidth (width/ratio)
		fig.set_figheight (width)
	plt.xlabel ('focal-plane x-position')
	plt.ylabel ('focal-plane y-position')
	if title is not None :
		plt.title (title)

	# YUK !!!!!!!!!!!!
	sdummy = plt.scatter (xdummy,ydummy,s=0,c=cdummy,facecolors='none') # NEEDED BY fig.colorbar()
	plt.clim (fmin,fmax)
	divider = make_axes_locatable(ax)
	cax = divider.new_horizontal (size='3%', pad=0.05, pack_start=False)
	fig.add_axes (cax)
	cb = fig.colorbar (sdummy, cax=cax)
	if scale :
		cb.ax.set_ylabel ('mean intensity')
	else :
		cb.ax.set_ylabel ('mean flux')

	if outfile is not None :
		plt.savefig (outfile)
	if show :
		plt.show ()

def spectra2fits (spectra, config=None, keywords=pyFU_default_keywords, formats=pyFU_default_formats, orig_header=None) :
	"""
	Produces a 2- or 3-dimensional IFU image from a series of extracted IFU spectra.
	The spectral tables are expected to have metadata corresponding to a particular fibre.
	"""
	# GET THE DATA-TO-IMAGE CONFIGURATION

	naxis = 2
	key = 'naxis'
	if info is not None and key in info :
		naxis = info[key]
	logging.info (f'naxis : {naxis}')

	naxis1 = 1000
	key = 'naxis1'
	if info is not None and key in info :
		naxis1 = info[key]
	logging.info (f'naxis1 : {naxis1}')

	naxis2 = 1000
	key = 'naxis2'
	if info is not None and key in info :
		naxis2 = info[key]
	logging.info (f'naxis2 : {naxis2}')

	naxis3 = None
	key = 'naxis3'
	if info is not None and key in info :
		naxis3 = info[key]
	logging.info (f'naxis3 : {naxis3}')

	waves = None
	key = 'waves'
	if config is not None and key in config :
		waves = config[key]
	logging.info (f'waves: {waves}')

	# GET THE FIBRES FOR ALL SPECTRA AND THE TOTAL SIZE LIMITS
	xmax = -np.inf
	xmin =  np.inf
	ymax = -np.inf
	ymin =  np.inf
	fibres = []
	for i in range(len(spectra)) :
		spectrum = spectra[i]
		meta = spectrum.meta
		fibre = Fibre(header=meta,keywords=keywords,formats=formats)	# ASSUME ALL INFO IS IN THE METADATA
		fibres.append(fibre)

		x,y = fibre.pos
		r = fibre.inner_diameter/2.
		if x-r < xmin : xmin = x-r
		if x+r > xmax : xmax = x+r
		if y-r < ymin : ymin = y-r
		if y+r > ymax : ymax = y+r

	# SET THE PLATESCALE
	xsize = xmax-xmin
	ysize = ymax-ymin
	if xsize/naxis1 < ysize/naxis2 :
		platescale = xsize/naxis1
	else :
		platescale = ysize/naxis2

	# CREATE A WCS
	w = WCS(naxis=naxis)

	# CREATE NEW IMAGE HDU
	if naxis == 3 :
		data = np.zeros((naxis3,naxis2,naxis1))
		if waves is None :
			wav = spectra[0]['wavelength']
			wave1 = wav[0]
			wave2 = wav[-1]
		else :
			wave1 = waves[0]
			wave2 = waves[1]
		dwave = (wave2-wave1)/(naxis3-1)
		w.wcs.crpix = [naxis1//2,naxis2//2,1.]
		w.wcs.cdelt = [platescale,platescale,dwave]
		w.wcs.crval = [0.,0.,wave1]
	else :
		data = np.zeros((naxis2,naxis1))
		w.wcs.crpix = [naxis1//2,naxis2//2]
		w.wcs.cdelt = [platescale,platescale]
		w.wcs.crval = [0.,0.]

	# FILL IN EACH FIBRE
	for i in range(len(spectra)) :
		spectrum = spectra[i]
		fibre = fibres[i]

		# GET WAVELENGTHS & INTEGRATED FLUXES
		wavelength = spectrum['wavelength']
		flux = spectrum['flux']
		flx = np.sum(flux)	# hist_integral (wavelength,flux,None,None,wave1,wave2)

		x,y = fibre.pos
		r = fibre.inner_diameter/2.
		if x is not None and y is not None and r > 0. :
			if naxis == 2 :
				pix = w.wcs_world2pix (np.array([[x,y]]),0)
				ic = int(pix[0][0])
				jc = int(pix[0][1])
				mask = in_circle (ic,jc,r*platescale,naxis1,naxis2)
				if np.size(mask) == 0 :
					logging.info (f'fibre #{idx} not added to IFU image')
				else :
					logging.info (f'adding fibre #{idx} to IFU image ...')
					for i,j in mask :
						data[j][i] = flx

	# PREPARE FITS HDU
	hdu = fits.PrimaryHDU (data)	# =data,header=orig_hdr)
	hdr = hdu.header

	# ADD SPECIAL KEYWORDS AND RETURN
	if waves is not None :
		if 'wave1' in keywords :
			hdr[keywords['wave1']] = waves[0]
		if 'wave2' in keywords :
			hdr[keywords['wave2']] = waves[1]
	return hdu

def in_circle (i,j,r,naxis1,naxis2) :
	""" List of positions centered at i,j within r pixels of center """
	c = []
	r2 = r*r
	for ii in range(i-r,i+r+1,1) :
		for jj in range(j-r,j+r+1,1) :
			rr = (ii-i)**2+(jj-j)**2
			if rr < r2 :
				if ii >= 0 and jj >= 0 and ii < naxis1 and jj < naxis2 :
					c.append((ii,jj)) 
	return c

def main () :
	import matplotlib.pyplot as plt
	import sys
	import yaml
	from pyFU.utils import parse_arguments, read_tables, initialize_logging

	# ---- GET DEFAULT CONFIGURATION AND COMMAND LINE PARAMETERS
	README = """
Script for converting a set of extracted spectra into a 2-dimensional IFU image.
	"""
	arguments = {
		'cmap':{'path':'image:','default':'viridis', \
				'flg':'-c','type':str,'help':'matplotlib colour map name'},
		'fudge':{'path':'image:','default':1.35, \
				'flg':'-F','type':float,'help':'fudge factor for setting the image window size'},
		'generic': {'path':None,
				'default':None,  'flg':'-G','type':str,'help':'YAML file for generic image configuration info'},
		'infiles':{'path':'image:','default':None, \
				'flg':'-i','type':str,'help':'input FITS table file(s)'},
		'info':{'path':'image:','default':[], \
				'flg':'-I','type':list,'help':'additional FITS keyword data to display in image'},
		'labeled':{'path':'image:','default':False, \
				'flg':'-l','type':bool,'help':'print fibre label on the fibre image'},
		'logscale':{'path':'image:','default':False, \
				'flg':'-L','type':bool,'help':'show IFU image using logarithmic intensity scaling'},
		'outfiles':{'path':'image:','default':None, \
				'flg':'-o','type':str,'help':'output image file(s) (fits,png,jpg,...)'},
		'pixels':{'path':'image:','default':None, \
				'flg':'-x','type':list,'help':'integration pixels of output image'},
		'plot':{'path':None,'default':False, \
				'flg':'-p','type':bool,'help':'display resulting image'},
		'scale':{'path':None,'default':False, \
				'flg':'-s','type':bool,'help':'scale fluxes by fibre area (show intensity, not flux)'},
		'width':{'path':None,'default':9., \
				'flg':'-w','type':float,'help':'width of plot [in]'},
		'waves':{'path':'image:','default':None, \
				'flg':'-W','type':list,'help':'integration wavelengths of output image'},
		'yaml':{'path':None,'default':None, \
				'flg':'-y','type':str,'help':'name of pyFU configuration file'}
		}
	args,cfg = parse_arguments (arguments)
	info = cfg['image']

	# ---- LOGGING
	initialize_logging (config=cfg)
	logging.info ('*************************** image ******************************')

	# ---- OUTPUT GENERIC CONFIGURATION FILE?
	if args.generic is not None :
		logging.info ('Appending generic calibration configuration info to'+str(args.generic))
		with open (args.generic,'a') as stream :
			yaml.dump ({'image':info}, stream)
		sys.exit(0)

	# ---- GET THE INPUT AND OUTPUT FILES
	infiles,outfiles = get_infiles_and_outfiles (args.infiles,args.outfiles,cfg=info)

	# ---- FOR ALL INPUT AND OUTPUT FILES
	for infile,outfile in zip(infiles,outfiles) :
		logging.info (f'Reading {infile} ...')
		spectra,header = read_tables (pathname=infile)
		nspectra = len(spectra)

		# INFO TO DISPLAY
		metadata = {}
		for key in info['info'] :
			if key in header :
				metadata[key] = header[key]

		if outfile is not None :
			logging.info ('Writing image to {outfile} ...')
			if outfile.endswith('.fits') :
				raise NotImplementedError ('FITS output not yet supported!')

		# CONVERT SPECTRA TO A matplotlib GRAPH AND HENCE TO A PIXEL IMAGE
		if 'waves' in info and info['waves'] is not None :
			w = info['waves']
			title = f'{0} (sum from {w[0]:.2f} to {w[1]:.2f} nm)'
			logging.info (f'Producing wavelength-bounded focal-plane image {outfile}')
			if info['logscale'] :
				logging.info ('... with logarithmic intensity scale ...')
			spectra2image (spectra, title=title, waves=w, outfile=outfile,
							show=args.plot, logscale=info['logscale'], info=metadata,
							fudge=info['fudge'], scale=args.scale, width=args.width,
							labeled=info['labeled'])

		elif 'pixels' in info and info['pixels'] is not None :
			p = info['pixels']
			title = f'{infile} (sum from {p[0]:.2f} to {p[1]:.2f} pix)'
			logging.info (f'Producing wavelength-bounded focal-plane image {outfile}')
			logging.info ('Producing pixel-bounded focal-plane image ...')
			spectra2image (spectra, title=title, pixels=info['pixels'], info=metadata,
							outfile=outfile, show=args.plot, logscale=info['logscale'],
							fudge=info['fudge'], scale=args.scale, width=args.width,
							labeled=info['labeled'])

		else :
			title = f'{infile} (mean)'
			logging.info (f'Producing mean focal-plane image {outfile}')
			spectra2image (spectra, title=title, outfile=outfile, info=metadata,
							show=args.plot, logscale=info['logscale'], fudge=info['fudge'],
							scale=args.scale, width=args.width,
							labeled=info['labeled'])

		logging.info ('****************************************************************\n')

if __name__ == '__main__' :
	main ()

