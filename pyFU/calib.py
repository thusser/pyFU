#!/usr/bin/env python3

# test/calib.py 

import enum
import matplotlib.pyplot as plt
import logging
import numpy as np
import os, sys

from astropy.io    import fits
from scipy.ndimage import median_filter
from pyFU.utils    import check_directories, construct_new_path, get_sec, get_list_of_paths_and_filenames, get_infiles_and_outfiles, vector2Table, is_number
from pyFU.defaults import pyFU_default_keywords


class SimplePipeline (object) :
	"""
	Simple object for reducing raw data.
	"""
	def __init__ (self, keywords=pyFU_default_keywords) :
		self.keywords = keywords
		self.master_bias = None
		self.unit_dark   = None
		self.master_flat = None

	def get_exposure_time (self, hdr, millisec=False) :
		"""
		Extracts exposure time from header.
		"""
		t = 1.
		if 'exptime' in self.keywords :
			keyw = self.keywords['exptime'][0]
		else :
			keyw = 'EXPTIME'
		try :
			t = hdr[keyw]
		except KeyError as e :
			logging.error ('unable to access keyword '+keyw)
		if millisec : t *= 1000
		return t

	def get_biassec (self, hdu, biassec=False) :
		"""
		Parses the argument "biassec" as being either bool (to extract from
		a FITS header or not) or a numpy range y1,y2,x1,x2 for using hdu.data[y1:y2,x1:x2].
		Returns the biassec value for the given HDU.

		Note that the FITS standard is, where x1,x2,y1,y2 are in numpy.array notation,

			[x1+1:x2,y1+1:y2]

		i.e. the order is reversed and the FITS coordinates are 1- rather than 0-based.
		"""
		b = 0
		raw = hdu.data
		hdr = hdu.header

		if biassec is not None :
			if isinstance(biassec,bool) :
				if biassec :
					if 'biassec' in self.keywords :
						keyw = self.keywords['biassec'][0]
					else :
						keyw = 'BIASSEC'
					if keyw not in hdr :
						logging.error ('no {keyw} key in FITS header')
					else :
						y1,y2,x1,x2 = get_sec (hdr,key=keyw)
						b = np.nanmedian(raw[y1:y2, x1:x2])
			elif isinstance(biassec,list) and len(biassec) == 4 :
				y1,y2,x1,x2 = biassec	# numpy RANGE COORDINATES
				b = np.nanmedian(raw[y1:y2, x1:x2])
			else :
				logging.error (f'cannot determine biassec from {biassec}')
		return b

	def bias_subtraction (self, hdu, biassec=None, bias=None) :
		"""
		Subtract biassec and/or bias from HDU data.
		Returns the the new bias-subtracted image and the value of the biassec that was subtracted.
		"""
		img = np.array(hdu.data,dtype=float)
		hdu.data = img
		hdr = hdu.header
		bs = 0

		# SUBTRACT BIASSEC
		if biassec is not None :
			bs = self.get_biassec (hdu,biassec=biassec)
			img -= bs
			hdr['comment'] = f'subtracted biassec={bs}'

		# SUBTRACT IMAGE/NUMBER?
		if bias is None :
			b = self.master_bias
		else :
			b = bias
		if b is not None :
			if isinstance(b,float) or isinstance(b,int) or b.shape == img.shape :
				hdr.data = img-b
				hdr['comment'] = f'subtracted bias'
			else :
				logging.error ('data and bias images have different shapes!')
		return hdr.data,bs

	def global_bias (self, imagelist, show=False, outfile=None, biassec=None, hdu0=0) :
		"""
		Returns the median bias from a list of (biassec-corrected?) bias images.
		"""
		if imagelist is None or len(imagelist) == 0 :
			logging.error ('no list of bias images')
			return None
		n = len(imagelist)
		logging.info ('calculating median bias from {0} input files...'.format(n))

		# READ FIRST IMAGE TO GET SIZE
		name = imagelist[0]
		hdus = fits.open (name)
		hdu = hdus[hdu0]
		ny,nx = hdu.data.shape
		header = hdu.header.copy()
		hdus.close()

		# PUT ALL DATA INTO AN ARRAY
		data = np.zeros ((n,ny,nx))
		comments = []
		for i in range(n) :
			l = imagelist[i]
			hdus = fits.open (l)
			hdu = hdus[hdu0]
			raw = hdu.data
			med,std = np.nanmedian(raw),np.nanstd(raw)

			# SUBTRACT BIASSEC
			if biassec is None :
				data[i,:,:] = raw
				c = f'{name} : med={med:.2f},std={std:.2f}'
			else :
				b = self.get_biassec (hdu,biassec=biassec)
				data[i,:,:] = raw-b
				c = f'{name} : biassec={b},med={med:.2f},std={std:.2f}'
			logging.info ('... '+c)
			comments.append(c)
			hdus.close()

		# GET THE MEDIAN
		self.master_bias = np.nanmedian (data,axis=0)
		med,std = np.nanmedian(self.master_bias),np.nanstd(self.master_bias)
		logging.info ('master bias: median={0:.2f}, std={1:.2f}'.format(med,std))

		# SHOW?
		if show :
			im = plt.imshow (self.master_bias, interpolation='none', origin='lower', vmin=med-3*std, vmax=med+3*std)
			plt.colorbar(im)
			plt.title ('master bias')
			plt.show ()

		# SAVE?
		if outfile is not None :
			logging.info ('writing master bias to {0} ...'.format(outfile))
			hdu = fits.PrimaryHDU (data=self.master_bias)
			hdr = hdu.header
			for c in comments :
				hdr['comment'] = c
			hdr['comment'] = 'median of {0} biases'.format(len(imagelist))
			hdr.extend (header,update=False)
			if not check_directories (outfile,create=True) :
				logging.error ('cannot create output file!')
			else :
				hdu.writeto (outfile,overwrite=True)

		return self.master_bias

	def global_unit_dark (self, imagelist, method='median', bias=None, subtract_bias=True, \
					biassec=None, show=False, outfile=None, millisec=False, hdu0=0) :
		"""
		Returns the unit dark frame from a list of dark images.
		"""
		if imagelist is None or len(imagelist) == 0 :
			logging.info ('no dark-current removal/creation wanted')
			return None
		n = len(imagelist)
		logging.info ('calculating unit dark from {0} input files...'.format(n))

		# GET SHAPE OF DATA
		name = imagelist[0]
		hdus = fits.open (name)
		hdu = hdus[hdu0]
		ny,nx = hdu.data.shape
		data = np.zeros ((n,ny,nx))
		header = hdu.header.copy()
		hdus.close()

		# PUT ALL DATA INTO AN ARRAY
		comments = []
		for i in range(n) :
			name = imagelist[i]
			hdus = fits.open (name)
			hdu = hdus[hdu0]
			hdr = hdu.header
			raw = hdu.data
			med,std = np.nanmedian(raw),np.nanstd(raw)

			# SUBTRACT BIAS
			if subtract_bias :
				raw,bs = self.bias_subtraction (hdu,biassec=biassec,bias=bias)	# RETURNS biassec

			# DIVIDE BY EXPOSURE TIME
			t = self.get_exposure_time (hdr,millisec=millisec)
			data[i,:,:] = raw/t
			hdus.close()
			if subtract_bias :
				s = f'{name} : biassec={bs},med={med:.2f},std={std:.2f},exptime={t:.3f}'
			else :
				s = f'{name} : med={med:.2f},std={std:.2f},exptime={t:.3f}'
			logging.info ('...'+s)
			comments.append(s)

		# GET THE UNIT DARK
		if method == 'median' :
			self.unit_dark = np.nanmedian (data,axis=0)
		else :
			self.unit_dark = np.nanmean (data,axis=0)
		med,std = np.nanmedian(self.unit_dark),np.nanstd(self.unit_dark)
		logging.info (f'unit dark: median={med:.2f}, std={std:.2f}')

		# SHOW?
		if show :
			im = plt.imshow (self.unit_dark, interpolation='none', origin='lower', vmin=med-3*std, vmax=med+3*std)
			plt.colorbar(im)
			plt.title ('unit dark')
			plt.show ()

		# SAVE?
		if outfile is not None :
			logging.info (f'writing unit dark to {outfile} ...')
			hdu = fits.PrimaryHDU (data=self.unit_dark)
			hdr = hdu.header
			hdr['EXPTIME'] = (1.0,'unit exposure time of 1 sec')
			for c in comments :
				hdr['comment'] = c
			hdr['comment'] = f'median of {len(imagelist)} unit darks'
			hdr.extend (header,update=False)
			if not check_directories (outfile,create=True) :
				logging.error ('cannot create output file!')
			else :
				hdu.writeto (outfile,overwrite=True)

		return self.unit_dark

	def global_flat (self, imagelist, bias=None, unitdark=None, subtract_bias=True, biassec=None, \
						subtract_dark=True, show=False, outfile=None, millisec=False, hdu0=0) :
		"""
		Returns the median scaled flatfield frame from a list of flatfield images.
		"""
		if imagelist is None or len(imagelist) == 0 :
			logging.error ('no list of flat images')
			return None
		n = len(imagelist)

		# GET SHAPE OF DATA
		name = imagelist[0]
		hdus = fits.open (name)
		hdu = hdus[hdu0]
		ny,nx = hdu.data.shape
		data = np.zeros ((n,ny,nx))
		header = hdu.header.copy()
		hdus.close()

		# PUT ALL DATA INTO AN ARRAY
		for i in range(n) :
			name = imagelist[i]
			hdus = fits.open (name)
			hdu = hdus[hdu0]
			raw = hdu.data
			hdr = hdu.header
			med,std = np.nanmedian(raw),np.nanstd(raw)

			# SUBTRACT BIAS
			if subtract_bias :
				unbiased,bs = self.bias_subtraction (hdu,biassec=biassec,bias=bias)	# RETURNS biassec

			# GET EXPOSURE TIME
			t = self.get_exposure_time (hdr,millisec=millisec)

			# GET UNIT DARK
			d = 0.
			if subtract_dark :
				if unitdark is None :
					d = self.unit_dark
				else :
					d = unitdark
				if d is None :
					logging.error ('no unit dark available to subtract from flat!')

			# LOG
			s = f'{name} : '
			if subtract_bias :
				s += 'bias-corr'
			if subtract_bias and biassec is not None :
				s += f',biassec={bs:.2f}'
			if subtract_dark :
				s += ',dark-corr'
			s += f' med={med:.2f},std={std:.2f},exptime={t:.3f}'
			logging.info ('...'+s)

			# CALIBRATE
			cal = unbiased-d*t	# (RAW-BIAS)-UNITDARK*EXPTIME
			calnorm = np.nanmedian(cal)

			# NORMALIZE
			data[i] = cal/calnorm

			hdus.close()

		# GET THE UNIT MEDIAN FLATFIELD
		self.master_flat = np.nanmedian (data,axis=0)
		med,std = np.nanmedian(self.master_flat),np.nanstd(self.master_flat)
		logging.info (f'master flat: median={med:.2f}, std={std:.2f}')

		# SHOW?
		if show :
			im = plt.imshow (self.master_flat, interpolation='none', origin='lower', vmin=med-3*std, vmax=med+3*std)
			plt.colorbar(im)
			plt.title ('master flat')
			plt.show ()

		# SAVE?
		if outfile is not None :
			logging.info (f'writing master flat to {outfile} ...')
			hdu = fits.PrimaryHDU (data=self.master_flat)
			hdr = hdu.header
			hdr['comment'] = f'median of {len(imagelist)} normalized flats'
			hdr.extend (header,update=False)
			if not check_directories (outfile,create=True) :
				logging.error ('cannot create output file!')
			else :
				try :
					hdu.writeto (outfile,overwrite=True)
				except e :
					logging.error (f'cannot writeto {outfile}: {str(e)}')
		return self.master_flat

	def calibrate (self, hdu, bias=None, unitdark=None, flat=None, subtract_bias=False, biassec=None, \
						subtract_dark=False, divide_flat=False, show=False, millisec=False, hdu0=0) :
		raw = hdu.data
		hdr = hdu.header
		ny,nx = raw.shape
		med,std = np.nanmedian(raw),np.nanstd(raw)
		s = f'raw: avg,std,exptime={med:.2f},{std:.2f},'

		# SUBTRACT BIAS
		if subtract_bias :
			bs = self.bias_subtraction (hdu,biassec=biassec,bias=bias)	# RETURNS biassec
			raw = hdu.data

		# GET EXPOSURE TIME
		t = self.get_exposure_time (hdr,millisec=millisec)
		s += f'{t:.3f}, '

		# GET UNIT DARK
		d = 0.
		if subtract_dark :
			if unitdark is None :
				d = self.unit_dark
			else :
				d = unitdark
			if d is None :
				logging.error ('no unit dark available to subtract from flat!')
			elif not isinstance(d,float) and not isinstance(d,int) and d.shape != raw.shape :
				logging.error ('data and dark images have different shapes!')
				return None

		# REMOVE DARK 
		cal = raw-d*t

		# GET FLAT
		f = 1.
		if divide_flat :
			if flat is None :
				f = self.master_flat
			else :
				f = flat
			if f is None :
				logging.error ('no flat to divide')
				return False
			hdr['comment'] = 'divided by flatfield'

		# CALIBRATE
		result = cal/f
		s += f'result: avg,std={np.nanmean(result):.2f},{np.nanstd(result):.2f}'
		logging.info (s)
		hdu.data = result

		# SHOW?
		if show :
			show_hdu (hdu)
			if 'FILENAME' in hdu.header :
				plt.title ('calibrated'+hdu.header['FILENAME'])
			plt.show ()

		return True 

	def maths (self, file1=None, oper=None, thing2=None, dataset=0) :
		"""
		Function for doing simple maths of the form "file1 + thing2"
		or "{function} thing2" with images.
		"dataset" is the index of the HDU in any HDU list.
		"""
		# GET data1 
		img1_used = False
		data1 = None
		if file1 is not None and '.fit' in file1 :	# GET data1 FROM HDU
			hdu1 = fits.open (file1)[dataset]
			data1 = np.array(hdu1.data,dtype=float)
			img1_used = True
		elif file1 is not None :						# GET float
			data1 = float(file1)

		# GET data2
		img2_used = False
		data2 = None
		if isinstance(thing2,float) :						# GET float
			if not img1_used :
				logging.error ('no image data in special operation')
				return None
			data2 = float(thing2)
		elif isinstance (thing2,fits.PrimaryHDU) :			# GET HDU DATA
			data2 = thing2.data
			img2_used = True
			hdu2 = thing2
		elif isinstance (thing2,str) and '.fit' in thing2 :	# GET DATA FROM FITS IMAGE
			hdu2 = fits.open (thing2)[dataset]
			data2 = np.array(hdu2.data,dtype=float)
			img2_used = True
		else :											# GET float
			logging.error ('maths argument is not number|HDU|filename')
			return None

		# PERFORM OPERATION file1 oper thing2 OR oper thing2
		if oper == '+' :
			data3 = data1+data2
		elif oper == '-' :
			if file1 is None :
				data3 = data1-data2
			else :
				data3 = -data2
		elif oper == '*' :
			data3 = data1*data2
		elif oper == '/' :
			data3 = data1/data2
		elif oper == '^' or oper == '**' :
			data3 = data1**data2
		elif file1 is None and oper == 'abs' :
			data3 = np.nanabs (data2)
		elif file1 is None and oper == 'mean' :
			data3 = np.nanmean (data2)
		elif file1 is None and oper == 'median' :
			data3 = np.nanmedian (data2)
		elif file1 is None and oper == 'sqrt' :
			data3 = np.sqrt (data2)
		elif file1 is None and oper == 'flatten' :
			data3 = data2/median_filter (data2, size=50)
		elif file1 is None and oper == 'xmean' :
			data3 = np.nanmean (data2,axis=0)
		elif file1 is None and oper == 'ymean' :
			data3 = np.nanmean (data2,axis=1)

		if oper == 'xmean' or oper == 'ymean' :			# RETURNS 1-D DATA, NOT IMAGE
			c1 = fits.Column (name='pixel',array=np.arange(len(data3)), format='K')
			c2 = fits.Column (name='flux', array=data3, format='K')
			hdu = fits.BinTableHDU.from_columns ([c1,c2], header=hdu2.header)
			hdu.header['comment'] = f'data: {oper} {thing2}'
			return hdu
		elif img1_used :
			hdu = fits.PrimaryHDU (data=data3,header=hdu1.header)		# GET COPY OF HEADER
			hdu.header['comment'] = f'data: {file1} {oper} {thing2}'
			return hdu
		elif img2_used :
			hdu = fits.PrimaryHDU (data=data3,header=hdu2.header)
			hdu.header['comment'] = f'data: {oper} {thing2}'
			return hdu
		else :
			logging.error ('should not be able to get here!')
			return None

def main () :
	import yaml
	from pyFU.utils import parse_arguments, initialize_logging
	from pyFU.display import show_hdu

	# ---- GET DEFAULTS AND PARSE COMMAND LINE
	arguments = {
		'abs': {'path':None,
			'default':False, 'flg':'-W','type':bool,'help':'abs value of input images'},
		'average': {'path':None,
			'default':False, 'flg':'-A','type':bool,'help':'average of input images'},
		'biassec': {'path':'calib:',
			'default':None,  'flg':'-x','type':str,'help':'boolean or y1,y2,x1,x2 (numpy range coords)'},
		'bias_files': {'path':'calib:bias:infiles',
			'default':None,  'flg':'-1','type':str,'help':'pattern for raw bias pathnames'},
		'dark_files': {'path':'calib:dark:infiles',
			'default':None,  'flg':'-2','type':str,'help':'pattern for raw dark pathnames'},
		'divide': {'path':None,
			'default':False, 'flg':'-Q','type':bool,'help':'divide the input images by the other images/number'},
		'divide_flat': {'path':'calib:flat:',
			'default':False, 'flg':'-F','type':bool,'help':'divide image by master flat'},
		'flatten': {'path':None,
			'default':False,  'flg':'-J','type':bool,'help':'flatten (for flatfield images)'},
		'flat_files': {'path':'calib:flat:infiles',
			'default':None,  'flg':'-3','type':str,'help':'pattern for raw flat pathnames'},
		'generic': {'path':None,
			'default':None,  'flg':'-G','type':str,'help':'YAML file for generic calib configuration info'},
		'infiles':    {'path':'calib:',
			'default':None,  'flg':'-i','type':str,'help':'name of FITS image files to process'},
		'masterbias': {'path':'calib:bias:',
			'default':None,  'flg':'-b','type':str,'help':'pathname of master bias image'},
		'masterflat': {'path':'calib:flat:',
			'default':None,  'flg':'-f','type':str,'help':'pathname of master flatfield image'},
		'millisec': {'path':None,
			'default':False, 'flg':'-m','type':bool,'help':'EXPTIME is in millisecs'},
		'minus': {'path':None,
			'default':False, 'flg':'-M','type':bool,'help':'subtract other images/number from input images'},
		'other':   {'path':None,
			'default':None,  'flg':'-O','type':str,'help':'pathname of other FITS image file'},
		'outfiles':   {'path':'calib:',
			'default':None,  'flg':'-o','type':str,'help':'pathname of output FITS image file'},
	    'plot':      {'path':None,
			'default':False, 'flg':'-p','type':bool,'help':'plot details'},
		'plus': {'path':None,
			'default':False, 'flg':'-P','type':bool,'help':'add other image to the input image'},
		'raised_by': {'path':None,
			'default':False, 'flg':'-^','type':bool,'help':'raise the input images by the other images/number'},
		'start_hdu': {'path':None,
			'default':0,     'flg':'-0','type':int,'help':'number of starting HDU in input files'},
		'sqrt_of': {'path':None,
			'default':False, 'flg':'-R','type':bool,'help':'sqrt of images'},
		'subtract_bias': {'path':'calib:bias:',
			'default':False, 'flg':'-B','type':bool,'help':'subtract master bias from image'},
		'subtract_dark': {'path':'calib:dark:',
			'default':False, 'flg':'-D','type':bool,'help':'subtract scaled unit dark from image'},
		'sum': {'path':None,
			'default':False, 'flg':'-S','type':bool,'help':'sum all of the input images'},
		'times': {'path':None,
			'default':False, 'flg':'-X','type':bool,'help':'multiply input images by the other images'},
		'trimsec': {'path':'calib:',
			'default':None,  'flg':'-T','type':str,'help':'boolean or y1,y2,x1,x2 (numpy range coords)'},
		'unitdark': {'path':'calib:dark:',
			'default':None,  'flg':'-d','type':str,'help':'pathname of unit dark image'},
		'xmean': {'path':None,
			'default':None,  'flg':'-_','type':bool,'help':'project along y'},
		'ymean': {'path':None,
			'default':None,  'flg':'-/','type':bool,'help':'project along x'},
		'yaml':      {'path':None,
			'default':None,  'flg':'-y','type':str,'help':'global YAML configuration file for parameters'}
		}
	args,cfg = parse_arguments (arguments)

	# ---- GET TOPIC DICTINARY
	info = cfg['calib']
	logging.debug ('\ncfg:\n'+str(info))

	# ---- LOGGING
	initialize_logging (config=cfg)
	logging.info ('********************* raw image pipeline / image manipulator **********************')

	# ---- OUTPUT GENERIC CONFIGURATION FILE?
	if args.generic is not None :
		logging.info ('Appending generic calibration configuration info to'+str(args.generic))
		with open (args.generic,'a') as stream :
			yaml.dump ({'calib':info}, stream)
		sys.exit(0)

	# ---- GET LISTS OF INPUT AND OUTPUT FILES
	infiles,outfiles = get_infiles_and_outfiles (args.infiles,args.outfiles)

	# ---- GET SIMPLE PIPELINE OBJECT
	pipel = SimplePipeline ()
	sub_bias    = False
	sub_dark    = False
	div_flat    = False
	use_biassec = False
	use_trimsec = False
	hdu         = None
	biassec     = False	# True IF biassec IS IN FITS HEADER
	trimsec     = False

	# ---- SPECIAL FUNCTIONS?
	special = args.sum or args.average or args.minus or args.plus or args.divide or args.times \
				or args.sqrt_of or args.xmean or args.ymean or args.raised_by or args.flatten
	if special :
		if args.subtract_bias or args.subtract_dark or args.divide_flat :
			logging.error ('special functions and bias/dark/flat manipulations do not mix!')
			sys.exit(1)

	# ---- CHECK FOR BIAS
	dbias = info['bias']
	create_bias = ('infiles'    in dbias) and (dbias['infiles']    is not None) and \
				  ('masterbias' in dbias) and (dbias['masterbias'] is not None)

	# SUBTRACT BIASSEC?
	if 'biassec' in info and info['biassec'] is not None :
		use_biassec = True
		biassec = info['biassec']

	# CREATE BIAS?
	if create_bias :
		logging.info ('creating master bias ...')
		dfiles = dbias['infiles']
		if isinstance (dfiles,str) :
			dfiles = get_list_of_paths_and_filenames (dfiles,mode='path')
		pipel.global_bias (dfiles, biassec=biassec, \
				show=args.plot, outfile=dbias['masterbias'], hdu0=args.start_hdu)

	# NEED BIAS IMAGE?
	if 'subtract_bias' in dbias and dbias['subtract_bias'] :
		sub_bias = True
		if pipel.master_bias is None :
			# GET BIAS FROM FILE
			if ('masterbias' in dbias) and (dbias['masterbias'] is not None) :
				bhdus = fits.open (dbias['masterbias'])
				pipel.master_bias = bhdus[args.start_hdu].data
			else :
				logging.error ('no master bias image given!')

	# ---- CHECK FOR UNIT DARK
	ddark = info['dark']
	create_dark = ('infiles'  in ddark) and (ddark['infiles']  is not None) and \
				  ('unitdark' in ddark) and (ddark['unitdark'] is not None)
	if create_dark :
		logging.info ('creating unit dark ...')
		dfiles = ddark['infiles']
		if isinstance (dfiles,str) :
			dfiles = get_list_of_paths_and_filenames (dfiles,mode='path')
		pipel.global_unit_dark (dfiles, show=args.plot, outfile=ddark['unitdark'], \
							biassec=biassec, millisec=cfg['millisec'], hdu0=args.start_hdu)

	# NEED DARK IMAGE?
	if 'subtract_dark' in ddark and ddark['subtract_dark'] :
		sub_dark = True
		if pipel.unit_dark is None :
			# GET DARK FROM FILE
			if ('unitdark' in ddark) and (ddark['unitdark'] is not None) :
				dhdus = fits.open (ddark['unitdark'])
				pipel.unit_dark = dhdus[args.start_hdu].data
			else :
				logging.error ('no unit dark image given!')

	# ---- CHECK FOR MASTER FLAT
	f = None
	dflat = info['flat']
	create_flat = ('infiles'    in dflat) and (dflat['infiles']    is not None) and \
				  ('masterflat' in dflat) and (dflat['masterflat'] is not None)
	if create_flat :
		logging.info ('creating master flat ...')
		ffiles = dflat['infiles']
		if isinstance (ffiles,str) :
			ffiles = get_list_of_paths_and_filenames (ffiles,mode='path')
		pipel.global_flat (ffiles, show=args.plot, outfile=dflat['masterflat'],
							biassec=biassec, bias=pipel.master_bias, millisec=cfg['millisec'], hdu0=args.start_hdu)

	# NEED FLAT IMAGE?
	if 'divide_flat' in dflat and dflat['divide_flat'] :
		div_flat = True
		if pipel.master_flat is None :
			# GET FLAT FROM FILE
			if ('masterflat' in dflat) and (dflat['masterflat'] is not None) :
				fhdus = fits.open (dflat['masterflat'])
				pipel.master_flat = fhdus[args.start_hdu].data
			else :
				logging.error ('no master flat image given!')

	# ---- GET OTHER DATA
	if args.other is not None :
		logging.info (f'other: {args.other}')
		if is_number (args.other) :
			other_data = float(args.other)
		else :
			other_data = fits.open (args.other)[args.start_hdu]

	# ---- GET TRIMSEC
	use_trimsec = 'trimsec' in info and info['trimsec'] is not None
	if use_trimsec :
		trimsec = info['trimsec']
		if isinstance(trimsec,bool) :	# trimsec BOOLEAN -> USE FITS HEADER
			if trimsec :
				if 'trimsec' in pipel.keywords :
					trimkey = pipel.keywords['trimsec'][0]
				else :
					trimkey = 'TRIMSEC'
			else :
				use_trimsec = False
				trimsec = None
		elif isinstance(trimsec,list) and len(trimsec) == 4 :	# trimsec A LIST -> y1,y2,x1,x2
			trimsec = [int(i) for i in trimsec]
		else :
			logging.error (f'trimse {trimsec} != y1,y2,x1,x2')

	# ---- CALIBRATE
	if (use_biassec or use_trimsec or sub_bias or sub_dark or div_flat) and (infiles is not None and outfiles is not None) :
		for infile,outfile in zip(infiles,outfiles) :
			s = ''
			if use_biassec : s += 'b'
			if sub_bias : s += 'B'
			if sub_dark : s += 'D'
			if div_flat : s += 'F'
			if use_trimsec : s += 't'
			logging.info (f'calibrating ({s}) {infile} ...')
			hdus = fits.open (infile)
			hdu  = hdus[args.start_hdu]
			hdr  = hdu.header

			# ---- REDUCE
			if not pipel.calibrate (hdu, subtract_bias=sub_bias, subtract_dark=sub_dark, biassec=biassec, \
										divide_flat=div_flat, millisec=cfg['millisec'], \
										hdu0=args.start_hdu) :
				logging.error ('could not calibrate image')
				sys.exit (1)

			# ---- TRIM
			if use_trimsec :
				if isinstance(trimsec,bool) :
					y1,y2,x1,x2 = get_sec (hdr,key=trimkey)
				elif isinstance(trimsec,list) :
					y1,y2,x1,x2 = trimsec
				hdu.data = hdu.data[y1:y2, x1:x2]
				s = '... trimmed to array[{0}:{1}, {2}:{3}]'.format(y1,y2,x1,x2)
				hdr['comment'] = s
				logging.info (s)

			# ---- PLOT
			if args.plot and hdu is not None :
				show_hdu (hdu)
				plt.title (outfile)
				plt.show ()

			# ---- SAVE RESULT
			logging.info (f'writing calibrated image to {outfile}')
			if not check_directories (outfile,create=True) :
				logging.error ('cannot create output file!')
			else :
				hdu.writeto (outfile,overwrite=True)
				outfiles.append (outfile)

	if special :
		# SIMPLE AVERAGE,SUM OF MULTIPLE FILES
		if args.sum or args.average :
			if len(infiles) != 1 and len(outfiles) <= 1 :
				logging.error ('cannot sum/average {0} images into {1} image'.format(len(outfiles),len(infiles)))
				sys.exit(1)
			hdu = None
			for filename in infiles :
				hdus = fits.open (filename)
				h  = hdus[args.start_hdu]
				if hdu is None :
					hdu = h
				else :
					hdu.data += h.data
					hdu.header['COMMENT'] = f'added {filename}'
			if args.average :
				hdr.data /= len(infiles)
				hdu.header['COMMENT'] = f'averaged by /{len(infiles)}'

			# ---- PLOT
			if args.plot and hdu is not None :
				show_hdu (hdu)
				plt.title (outfile)
				plt.show ()

			# ---- SAVE RESULT
			logging.info (f'writing calibrated image to {outfile}')
			if not check_directories (outfile,create=True) :
				logging.error ('cannot create output file!')
			else :
				hdu.writeto (outfile,overwrite=True)

		# ---- SPECIAL FUNCTIONS WITH TWO DATA ARGUMENTS, THE 2ND BEING THE OTHER DATA
		elif (args.minus or args.plus or args.divide or args.times or args.raised_by) \
					and (args.other is not None) \
					and (len(infiles) == len(outfiles)) :
			oper = None
			if args.plus      : oper = '+'
			if args.minus     : oper = '-'
			if args.divide    : oper = '/'
			if args.times     : oper = '*'
			if args.raised_by : oper = '^'
			print (infiles,outfiles)
			for infile,outfile in zip((infiles,outfiles)) :
				logging.info (f'{outfile} = {infile} {oper} {args.other}')
				hdu = pipel.maths (infile,oper,other_data)
				if hdu is not None :
					hdu.writeto (outfile,overwrite=True)

		# ---- SPECIAL SINGLE-ARGUMENT FUNCTIONS
		elif (args.xmean or args.ymean or args.sqrt_of or args.abs or args.flatten) \
					and len(infiles) == len(outfiles) :
			if args.xmean   : oper = 'xmean'
			if args.ymean   : oper = 'ymean'
			if args.sqrt_of : oper = 'sqrt'
			if args.abs     : oper = 'abs'
			if args.flatten : oper = 'flatten'
			for infile,outfile in zip(infiles,outfiles) :
				logging.info (f'{outfile} = {oper} {infile}')
				hdu = pipel.maths (None,oper,infile)
				if hdu is not None :
					hdu.writeto (outfile,overwrite=True)

		else :
			logging.error (f'cannot perform the special function')

		logging.info ('************************************************************************************\n')
				
if __name__ == '__main__' :
	main ()
