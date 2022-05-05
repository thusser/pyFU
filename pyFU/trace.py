#!/usr/bin/env python3

# pyFU/trace.py

"""
SpectrumTracer takes a well-exposed IFU flatfield (or similarly well-defined) image and traces
where the spectra are and how the relative amplitude between the spectra varies.
The dispersion direction is assumed to be roughly horizontal.  The number
and rough width of the individual spectra are assumed to be known.
"""

import enum
import numpy as np
import sys
import matplotlib as mpl
import logging
import yaml

from .utils      import centroid1D, find_peaks, show_hdu
from astropy.io import fits
from matplotlib import pyplot as plt

from scipy.optimize  import curve_fit
from scipy           import ndimage as ndi
# from skimage.feature import peak_local_max

from astropy.table   import Table

from pyFU.defaults import pyFU_default_keywords, pyFU_default_formats
from pyFU.display  import show_with_menu
from pyFU.ifu      import get_fibres, Fibre
from pyFU.meta     import header2config
from pyFU.utils    import UTC_now, merge_dictionaries, polynomial_functions, multiple_gauss_function
from pyFU.mgauss   import MultipleGaussians as MultiG

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level,format=pyFU_logging_format)


class SpectrumTracer (object) :
	def __init__ (self, hdu, config=None, keywords=None, formats=None, external_traces=None, \
						ignore=False) :
		"""
		Initialize the tracer, optionally with a configuration dictionary with default entries
		"""
		self.config = config
		self.hdu = hdu
		self.shape = None
		ny,nx = 0,0

		if hdu is not None :
			hdr = hdu.header
			self.shape = hdu.data.shape
			ny,nx = self.shape

		if keywords is None :
			self.keywords = dict(pyFU_default_keywords)
		else :
			self.keywords = keywords

		if formats is None :
			self.formats = dict(pyFU_default_formats)
		else :
			self.formats = formats

		self._fibres         = None
		self._vert_slices    = None
		self._x              = None
		self.sigma_factor    = 1.
		self.sigma_kappa     = 1.
		self.sigma_order     = 3
		self._sigma_fits     = None
		self._sigma_factors  = None
		self.ampl_order      = 5
		self.ampl_model      = 'polynomial'		# OR 'gaussians'
		self.bkg_factor      = 0.2
		self.trace_bias      = 0.
		self.spacing         = 6.
		self.dy_max          = 4.
		self.dy_min          = -4.
		self.number_slices   = 30
		self.number_fibres   = None
		self.number_traces   = None
		self.trace_order     = 5
		self.mid_slice       = None
		self.window_max      = 5
		self.window_centroid = 5
		self.window_profile  = 7
		self.ampl_format     = self.formats['ampl_format']
		self.trace_format    = self.formats['trace_format']
		self.x_min           = 0
		self.x_max           = nx
		self.external_slices = None

		# PARSE EXTERNAL TRACE FILE (YAML)
		if external_traces is not None :
			with open(external_traces) as stream :
				d = yaml.safe_load(stream)
			merge_dictionaries (self.config,d)

		# PARSE FITS HEADER/CONFIGURATION DICTIONARY
		self._parse_info (ignore=ignore)

	def _parse_info (self, ignore=False) :
		"""
		Reads configuration parameters from the stored configuration
		dictionary or from the header of the stored image HDU.
		"""
		# ---- GET CONFIGURATION DICTIONARY FROM HEADER
		if self.hdu is not None and not ignore :
			hcfg = header2config (self.hdu.header,keywords=self.keywords,formats=self.formats)
			if self.config is None :
				self.config = hcfg
			else :
				merge_dictionaries (self.config,hcfg)

		# ---- OR GET FROM CONFIGURATION FILE
		config = self.config
		info = {}

		# ---- UPDATE FORMAT INFO TO MATCH YAML CONFIGURATION
		if 'formats' in config :
			info = config['formats']
			merge_dictionaries (self.formats,info)

		# ---- BASIC IFU INFO...
		if 'ifu' in config :
			info = config['ifu']

			# ... GET NUMBER OF SPECTRA 
			key = 'number_fibres'
			if key in info :
				self.number_fibres = info[key]
				# self.number_traces = self.number_fibres

			# ... GET FIBRES
			if 'slit_labels' in info and info['slit_labels'] is not None :
				logging.info (f'getting fibres using slit_labels {info["slit_labels"]}')
				self._fibres = get_fibres (config=self.config, keywords=self.keywords, formats=self.formats)
				if self._fibres is None or len(self._fibres) == 0 :
					logging.error ('unable to reconstruct fibres!')
					return False

		# ---- TRACE INFO
		if config is not None and 'trace' in config :
			info = config['trace']

			# ... GET NUMBER OF TRACES
			# if self.number_traces is None and 'number_traces' in info :
			#	self.number_traces = info['number_traces']
			logging.info ('number_traces : {0}'.format(self.number_of_traces()))

			# ... GET VARIOUS PARAMETERS
			somethings = ['trace_order','trace_format', \
				'ampl_order','ampl_format', 'ampl_model', \
				'spacing','number_slices','window_max','window_centroid', \
				'dy_min','dy_max','bkg_factor','trace_bias','mid_slice','window_profile',\
				'sigma_factor','sigma_kappa','sigma_order','x_min','x_max']
			for key in somethings :
				if key in info :
					self.__dict__[key] = info[key]
				logging.info ('{0} : {1}'.format(key,self.__dict__[key]))
				if self.spacing is not None and self.dy_min is None :
					self.dy_min = -1.5*self.spacing
				if self.spacing is not None and self.dy_max is None :
					self.dy_max = 1.5*self.spacing

		# ---- CLEAN UP
		if self.mid_slice is None :
			self.mid_slice = self.number_slices//2
		if self.spacing is None :
			self.spacing = 6.54321
			logging.info ('assuming spacing of {0}'.format(self.spacing))
		if self.dy_min is None :
			self.dy_min = -1.5*self.spacing
		if self.dy_max is None :
			self.dy_max = 1.5*self.spacing
		logging.info (f'spacing={self.spacing},dy_min={self.dy_min},dy_max={self.dy_max}')

		if self.number_fibres is None :
			self.number_fibres = self.number_traces
			logging.info ('assuming number_fibres = number_traces')
		else :
			logging.info ('number_fibres : {0}'.format(self.number_fibres))
		return True

	def _find_separated_peaks (self, maxima) :
		"""
		Returns sorted list of maxima with reasonable separations.
		"""
		sepa = int(self.spacing)
		maxi = np.sort(maxima.reshape(-1))	# RAW LIST OF MAXIMA
		peaks = [maxi[0]]
		npeaks = 1
		for i in range(1,len(maxi)) :
			p = peaks[npeaks-1]
			if maxi[i]-p < sepa :
				peaks[npeaks-1] = int(0.5*(p+maxi[i])+0.5)
			else :
				peaks.append(maxi[i])
				npeaks += 1
		return peaks

	def use_external_slices (self, tab) :
		"""
		Use a Table of external slice positions e.g. determined from an arclamp image.
		Must have keys 'xavg','xleft','dx'.
		"""
		names = tab.colnames
		keys = ['xavg','xleft','dx']
		for key in keys :
			if key not in names :
				raise ValueError ('{0} not a column in external slice Table! Need keys {1}'.format(key,str(keys)))
		self.external_slices = tab
		self.number_slices = len(tab)
		self.mid_slice  = self.number_slices//2

	def get_slice_positions (self) :
		"""
		Returns a list of (xavg,xleft,dx) coordinates which define the vertical slices.
		Normally, these are simply evenly spaced, starting from the middle, but
		they can also be given by an external list.
		"""
		slice_pos = []
		if self.external_slices is None :
			ny,nx = self.shape
			dx = int((self.x_max-self.x_min)/self.number_slices)
			for k in range(self.number_slices) :
				i = self.x_min+(k+1)*dx		# i IS LAST ROW USED IN SLICE
				xavg = 0.5*(i-dx+i+1)
				xleft = i-dx
				xright = i+1
				if xleft < 0 : xleft=0
				if xright > nx : xright=nx
				slice_pos.append ([xavg,xleft,dx])
		else :
			for i in range(len(self.external_slices)) :
				row = self.external_slices[i]
				slice_pos.append ([row['xavg'],row['xleft'],row['dx']])
		return slice_pos

	def find_maxima (self, show=False) -> bool :
		"""
		Finds the maxima in a set of binned vertical segements of an image.
		The weighted centroids of the local maxima are also calculated.
		The resulting dictionaries are stored in the self._vert_slices list.
		"""
		self._vert_slices = []
		ny,nx = self.shape
		y = np.arange(ny)

		if show :
			figsize = (14,4)	# INITIAL GUESS (INCHES!)
			mpl.rcParams['figure.subplot.left']   = 0.10
			mpl.rcParams['figure.subplot.bottom'] = 0.15
			mpl.rcParams['figure.subplot.right']  = 0.98
			mpl.rcParams['figure.subplot.top']    = 0.90

		slice_pos = self.get_slice_positions ()

		# FOR EVERY SLICE
		for slcpos in slice_pos :
			xavg,x1,dx = slcpos

			# GET VERTICAL SUB-IMAGE
			subdata = self.hdu.data[:,x1:x1+dx]
			vslice = np.median (subdata,axis=1)		# TRY TO IGNORE COSMICS
			vslice -= np.min(vslice)
			limit = self.trace_bias+self.bkg_factor*0.5*(np.median(vslice)+np.mean(vslice))
			mask = vslice > limit

			# FIND PEAKS AS MAXIMA AND CENTROIDS
			# peaks = peak_local_max (vslice*mask, min_distance=self.window_max//3)
			# peaks = peak_local_max_1D (vslice*mask, min_distance=self.window_max//3)
			peaks = find_peaks (vslice*mask, w=self.window_max)
			logging.info ('{0} peaks detected @ xavg={1}'.format(len(peaks),xavg))
			if peaks is not None and len(peaks) > 0 :
				coords = np.array(peaks+0.5,dtype=int)	# self._find_separated_peaks (peaks)
				logging.info ('{0} peaks found @ xavg={1}'.format(len(coords),xavg))
				d = {'xavg':xavg,'ymaxs':coords}
				yavgs = []
				ywids = []
				logging.debug (f'finding peak pos,avg,wid: window={self.window_centroid}')
				for y in coords :
					yavg,ywid = centroid1D (vslice,y,self.window_centroid, get_sigma=False, get_fwhm=True, subt_bkg=False)	# NEEDED ???????
					# yavg,ywid = centroid1D (vslice,yavg,self.window_centroid, get_sigma=False, get_fwhm=True, subt_bkg=False)
					yavgs.append(yavg)
					ywids.append(ywid)
					logging.debug (f'\t{y},{yavg},{ywid}')
				d['yavgs'] = yavgs
				d['ywids'] = ywids
				d['data']  = vslice
				d['amps']  = vslice[coords]
				logging.debug ('yavgs:'+str(yavgs))
				logging.debug ('ywids:'+str(ywids))
				self._get_sigma_factors (d)
				self._vert_slices.append(d)

			else :
				logging.info ('no peaks found @ xavg={0}'.format(xavg))

			# PLOT THE SLICE AND THE FOUND MAXIMA
			if show :
				y = np.arange(ny)
				figsize = (14,4)	# INITIAL GUESS (INCHES!)
				fig = plt.figure (figsize=figsize)
				plt.style.use ('ggplot')
				plt.plot (y,vslice,'-',drawstyle='steps-mid',label='slice',color='black')
				if peaks is not None :
					plt.plot (yavgs,vslice[coords]-1,'+',color='red',label='peaks')
					h = 0.5*np.array(vslice[coords]-1)
					xerr = 0.5*np.array(ywids)
					plt.errorbar (yavgs,h,xerr=xerr,fmt='+',color='blue',label='FWHM')
				plt.plot (y,np.zeros(ny)+limit,'-.',color='gray',label='limit')
				plt.xlabel ('y [pix]')
				plt.ylabel ('slice')
				plt.title ('vertical slice #{0} : {1} peaks found'.format(xavg,len(coords)))
				plt.legend (bbox_to_anchor=(1.02,1), loc='upper left', ncol=1, fontsize='x-small')
				plt.tight_layout ()
				plt.show ()

		# SUCCESS OF A MINIMUM NUMBER OF SLICES PRODUCED MAXIMA
		if len(self._vert_slices) < 2 :
			return False
		else :
			return True

	def multigauss (self, x, *p) :
		"""
		A multi-Gauss function without a background and with a common sigma:

			SUM_j [ amp[j]*exp(-0.5*(x-pos[j])**2/sigma) ]

		The n = 2*ng+2 parameters are
			p[0]..p[np//2-2]	:	positions
			p[np//2]..p[-2]		:	amplitudes
			p[-1]				:   common sigma, optionally (odd number of parameters)

		If sigma is not fit, then it is calculated from self._sigma_fits and self._x.
		"""
		n = len(p)
		ng = n//2

		# GET PARAMETERS
		pos = p[0:ng]
		amp = p[ng:2*ng+1]

		# GET SIGMA
		if n%ng == 1 :
			sigma = p[-1]
		elif self._sigma_fits is None :
			raise ValueError ('multiguass: no sigma_fits available!')
		elif self.sigma_order <= 5 :
			f = polynomial_functions[min(5,self.sigma_order)]
			sigma = f(self._x,*self._sigma_fits)

		# CORRECT FOR DIFFERENT SIZED FIBRES
		if self._sigma_factors is not None :
			sigmas = self._sigma_factors*sigma
		else :
			sigmas = np.zeros(ng)+sigma

		# ADD UP ALL GAUSSIANS
		mg = x-x
		for i in range(ng) :
			ai = amp[i]
			xi = pos[i]
			mg += ai*np.exp(-0.5*(x-xi)**2/sigmas[i]**2)
		return mg+self.trace_bias

	def _get_sigma_factors (self, slice) :
		"""
		Produces the sigma_factors used to fit vertical slice profiles using
		a single fitted sigma for the particular vertical slice "islice".
		Two different fibre sizes are supported.
		"""
		# GET SLICE INFO
		ywids = slice['ywids']
		nwids = len(ywids)
		mwids = np.median(ywids)

		# GET RELATIVE WIDTHS
		factors = np.zeros(nwids)
		for i in range(nwids) :
			f = ywids[i]/mwids
			factors[i] = max(f,1.)
		logging.debug ('raw sigma factors: '+str(factors))

		# GET MEDIAN BASIC WIDTH AND OPTIONAL ADDITIONAL WIDTH
		sfact = np.std(factors)
		mfact = np.median(factors)
		logging.debug ('median sigma factor, stddev: {0},{1}'.format(mfact,sfact))

		mask1 = np.where((factors-1.) <  sfact*self.sigma_kappa)
		if len(mask1) == nwids :
			factors = np.zeros(nwids)+1.
		else :
			mask2 = np.where((factors-1.) >= sfact*self.sigma_kappa)
			if self.sigma_factor is None :
				factor = np.median(factors[mask2])/np.median(factors[mask1])
			else :
				factor = self.sigma_factor
			factors[mask1] = 1.0
			factors[mask2] = factor

		# SAVE WIDTH FACTORS
		slice['sigma_factors'] = factors
		logging.debug ('sigma_factors: '+str(factors))

	def fit_profiles (self, show=False, details=False) ->  bool :
		"""
		Fit the vertical slices with multiple Gaussians.
		"""
		logging.info ('fitting profiles ...')
		showing = show
		detailed = details

		if self._vert_slices is None :
			logging.error ('no vertical slices to fit')
			return False
		nslices = len(self._vert_slices)

		# FIT MID-SLICE FIRST TO GET ESTIMATE FOR sigma
		logging.info ('fitting sigma of slice #{0}...'.format(self.mid_slice))
		sigma0,showing,detailed = self._fit_profile (self.mid_slice, self.window_max/2., show=showing, details=detailed)
		vslice = self._vert_slices[self.mid_slice]
		if 'err_sigma' in vslice :
			logging.info ('... sigma={0:.3f} +/- {1:.3f}'.format(sigma0,vslice['err_sigma']))
		else :
			logging.info ('... no fit for slice #{0} !'.format(self.mid_slice))

		# FIT RIGHT SLICES
		sigma = sigma0
		for i in range(self.mid_slice+1,nslices) :
			logging.info ('fitting sigma of slice #{0}...'.format(i))
			sigma,showing,detailed = self._fit_profile (i, sigma, show=showing, details=detailed)
			vslice = self._vert_slices[i]
			if 'err_sigma' in vslice :
				logging.info ('... sigma={0:.3f} +/- {1:.3f}'.format(sigma,vslice['err_sigma']))
			else :
				logging.info ('... no fit for slice #{0} !'.format(i))

		# FIT LEFT SLICES
		sigma = sigma0
		for i in range(self.mid_slice-1,-1,-1) :
			logging.info ('fitting sigma of slice #{0}...'.format(i))
			sigma,showing,detailed = self._fit_profile (i, sigma, show=showing, details=detailed)
			vslice = self._vert_slices[i]
			if 'err_sigma' in vslice :
				logging.info ('... sigma={0:.3f} +/- {1:.3f}'.format(sigma,vslice['err_sigma']))
			else :
				logging.info ('... no fit for slice #{0} !'.format(i))

		# FIT sigmas
		func = polynomial_functions[self.sigma_order]
		x = []
		sigmas = []
		err_sigmas = []
		for vslice in self._vert_slices :
			if 'err_sigma' in vslice :
				x.append(vslice['xavg'])
				sigmas.append(vslice['sigma'])
				err_sigmas.append(vslice['err_sigma'])
		x          = np.array(x)
		sigmas     = np.array(sigmas)
		err_sigmas = np.array(err_sigmas)

		logging.info ('fitting sigmas...')
		mask = None
		smed = np.median(sigmas)
		sstd = np.std(sigmas)
		swid = np.median(np.abs(sigmas-smed))
		err_sig = np.sqrt(err_sigmas**2+swid**2)

		# try :
		if True :
			try :
				p,cov = curve_fit (func,x,sigmas,sigma=err_sig,maxfev=10000)
				self._sigma_fits = list(p)
				pold = p

				# MAKE sigma FIT ROBUST BY RE-WEIGHTING BAD POINTS
				badness = np.abs(sigmas-func(np.array(x),*p))/err_sig
				bmin,bmid,bmax = np.min(badness),np.mean(badness),np.max(badness)
				logging.debug ('min,mean,max sigma badness: {0},{1},{2}'.format(bmin,bmid,bmax))
				mask1 = badness > 1.5*bmid
				mask2 = err_sigmas*badness > 1.5*swid
				mask  = np.array(np.where (mask1+mask2)[0],dtype=int)
				if len(mask) > 0 :
					logging.info ('re-fitting sigmas...')
					err_sig[mask] *= 1.+np.abs(badness[mask]-1.)
					p,cov = curve_fit (func,x,sigmas,sigma=err_sig,maxfev=10000,p0=p)
				self._sigma_fits = list(p)
			except Exception as e :
				logging.warning ('fitting sigmas did not converge: {0}'.format(str(e)))
				self._sigma_fits = None
				func = None

		if show :
			x = np.array(x)
			xx = np.linspace(x[0],x[-1],500)
			plt.title ('Fit to spatial widths of spectra')
			plt.xlabel ('x [pix]')
			plt.ylabel ('Gaussian sigma [pix]')
			plt.errorbar (xx,xx-xx+smed,yerr=xx-xx+sstd,fmt='-',color='green',alpha=0.1)
			plt.errorbar (x,sigmas,yerr=err_sigmas,fmt='o',color='black')
			if mask is not None and np.sum(mask) > 0 :
				for i in mask :
					plt.errorbar ([x[i]],[sigmas[i]],yerr=[err_sigmas[i]],fmt='o',color='red')
			if func is not None :
				plt.plot (xx,func(xx,*p),'-',color='blue',label='final')
				plt.plot (xx,func(xx,*pold),'--',color='blue',label='initial')
			plt.legend (bbox_to_anchor=(1.02,1), loc='upper left', ncol=1, fontsize='x-small')
			plt.tight_layout ()
			plt.show ()

		# RE-FIT SLICES WITH FITTED sigmas
		showing = show
		detailed = details
		for i in range(nslices) :
			logging.info ('fitting slice #{0}...'.format(i))
			vslice = self._vert_slices[i]
			self._x = vslice['xavg']				# USED BY multigauss IF sigma=None
			sig,showing,detailed = self._fit_profile (i, None, show=showing, details=detailed)

		return True

	def _fit_profile (self, islice, sigma, show=False, details=False) :
		"""
		Fits a single vertical slice with multiple Gaussians.
		Returns the fitted sigma of the profile and the current status of the show and details flags.
		"""
		vslice  = self._vert_slices[islice]
		self._sigma_factors = vslice['sigma_factors']

		ny,nx = self.shape
		xavg    = vslice['xavg']
		vdata   = np.array(vslice['data'])
		yavgs   = np.array(vslice['yavgs'])
		ymaxs   = np.array(vslice['ymaxs'],dtype=int)
		y = np.arange(ny,dtype=float)

		# GET REGION TO FIT
		jlast = int(yavgs[-1]+self.spacing/2+0.5)
		if jlast > ny : jlast=ny

		j1 = int(np.min(ymaxs)-self.spacing)
		j2 = int(np.max(ymaxs)+self.spacing)
		if j1 <      0 : j1=0
		if j2 >= jlast : j2=jlast-1
		nj = j2-j1+1

		# GET SUB-DATA	
		ysub =     y[j1:j2+1]
		vsub = vdata[j1:j2+1]
		amps = vdata[ ymaxs ]

		# IF THINGS DON'T MATCH, EXIT
		if len(yavgs) != len(amps) :
			return np.nan,show,details

		# FIT X-DEPENDENCE OF PROFILES
		if sigma is None :
			npars = 2*len(amps)
			pars = np.concatenate((yavgs,amps))
		else :
			npars = 2*len(amps)+1
			pars = np.concatenate((yavgs,amps,[sigma]))

		# PLOT SLICE
		if details and sigma is not None :
			vfit = self.multigauss (y,*pars)
			figsize = (14,4)	# INITIAL GUESS (INCHES!)
			fig = plt.figure (figsize=figsize)
			plt.style.use ('ggplot')
			plt.plot (y[:jlast],vdata[:jlast],'-',label='slice',drawstyle='steps-mid',color='black')
			plt.plot (y[:jlast],vfit[:jlast],'-',label='fit',drawstyle='steps-mid',color='red')
			plt.xlabel ('y [pix]')
			plt.ylabel ('slice')
			plt.title ('pre-fit to vertical slice #{0:.2f}: {1:.2f}'.format(xavg,sigma))
			plt.legend (bbox_to_anchor=(1.02,1), loc='upper left', ncol=1, fontsize='x-small')
			plt.tight_layout ()
			reslt = show_with_menu (fig,['no more slice plots','ABORT'])
			if reslt == 'no more slice plots' :
				show = False
				details = False
			elif reslt == 'ABORT' :
				return np.nan,False,False

		try :
			pars,cov = curve_fit (self.multigauss,ysub,vsub,p0=pars,maxfev=20000)
			if sigma is not None :
				sigma = pars[-1]
				err_sigma = np.sqrt(cov[-1][-1])

			# PLOT SLICE
			if show and sigma is not None :
				vfit = self.multigauss (y,*pars)
				figsize = (14,4)	# INITIAL GUESS (INCHES!)
				fig = plt.figure (figsize=figsize)
				plt.style.use ('ggplot')
				plt.plot (y[:jlast],vdata[:jlast],'-',label='slice',drawstyle='steps-mid',color='black')
				plt.plot (y[:jlast],vfit[:jlast],'-',label='fit',drawstyle='steps-mid',color='red')
				plt.xlabel ('y [pix]')
				plt.ylabel ('slice')
				plt.title ('fit to vertical slice #{0:.2f}: {1:.2f}'.format(xavg,sigma))
				plt.legend (bbox_to_anchor=(1.02,1), loc='upper left', ncol=1, fontsize='x-small')
				plt.tight_layout ()
				reslt = show_with_menu (fig,['no more slice plots','ABORT'])
				if reslt == 'no more slice plots' :
					show = False
					details = False
				elif reslt == 'ABORT' :
					return np.nan,False,False

			# NOTE RESULTS OF FIT
			if sigma is None :
				vslice['yfit'] = pars[:npars//2]
				vslice['amps'] = pars[npars//2:]
			if sigma is not None :
				vslice['yfit']  = pars[:npars//2]
				vslice['amps']  = pars[npars//2:-2]
				vslice['sigma'] = sigma
				vslice['err_sigma'] = err_sigma
			return sigma,show,details

		except RuntimeError as e :
			return None,show,details

	def trace_spectra (self, show=False) -> bool :
		"""
		Uses the fitted maxima of the vertical traces to construct
		horizontal traces that follow individual spectra.
		"""
		if self._vert_slices is None or len(self._vert_slices) < 2 :
			logging.error ('Unsuitable trace.')
			return False
		nslices = len(self._vert_slices)
		if self.mid_slice is None :
			self.mid_slice = nslices//2

		# GET STARTING SLICE
		vslice1 = self._vert_slices[self.mid_slice]
		xavg1    = vslice1['xavg']
		yavgs1   = vslice1['yavgs']
		if 'yfit' in vslice1 :
			yavgs1 = vslice1['yfit']
		if 'sigma_factors' not in vslice1 :
			logging.error ('no sigma_factors in mid-slice?')
		else :
			factors1 = vslice1['sigma_factors']
		amps1 = vslice1['amps']
		amp0 = 1.	# np.nanmean(amps1)
		n1     = len(yavgs1)		# MAXIMUM NUMBER OF POSSIBLE FOUND SPECTRA
		vslice1['ids'] = range(n1)

		# INITIAL DICTIONARY OF IDENTIFIED SPECTRA
		spectra = {}
		for i in range(len(yavgs1)) :
			spectra[i] = {
				'x':[xavg1],
				'y':[yavgs1[i]],
				'index':i,
				'sigma_factors':[factors1[i]],
				'amplitudes':[amps1[i]]
				}

		# MATCH EVERY VERTICAL SLICE TO THE RIGHT OF MIDSLICE
		yoffset = n1*[0]	# INITIAL OFFSET IS 0
		for i in range(self.mid_slice+1,nslices) :
			logging.debug ('slice #{0} with offset {1}'.format(i,yoffset))
			vslice = self._vert_slices[i]
			yoffset = self._get_trace (vslice,spectra,-1,yoffset)

		# MATCH EVERY VERTICAL SLICE TO THE LEFT OF MIDSLICE
		yoffset = n1*[0]	# RESET OFFSETS
		ilast = 0
		for i in range(self.mid_slice-1,-1,-1) :
			logging.debug ('slice #{0} with offset {1}'.format(i,yoffset))
			vslice = self._vert_slices[i]
			yoffset = self._get_trace (vslice,spectra,ilast,yoffset)
			ilast = -1

		# SORT THE x VALUES
		yavgs = n1*[0]
		for i in range(n1) :
			d = spectra[i]
			x = np.array(d['x'])
			y = np.array(d['y'])
			f = np.array(d['sigma_factors'])		# ADDED BY _get_trace
			a = np.array(d['amplitudes'])
			mask = np.argsort(x)
			d['x']             = x[mask]
			d['y']             = y[mask]
			d['sigma_factors'] = f[mask]
			d['amplitudes']    = a[mask]/amp0
			yavgs[i] = np.mean(y)

		# SORT THE INDICES BY y VALUES
		mask = np.argsort(yavgs)
		if self._fibres is None or len(self._fibres) == 0 :
			logging.debug (f'resetting inner fibre list ({n1})')
			self._fibres = {}
		for idx in range(1,n1+1) :		# idx RUNS FROM 1 TO n1
			d = spectra[mask[idx-1]]
			d['index'] = idx
			d['sigma_factor'] = np.median(d['sigma_factors'])
			fibre = self.get_fibre (idx)
			if fibre is None :
				fibre = Fibre(idx=idx, config=self.config)
				logging.info (f'index {idx}-1 <= 0 or >= {len(self._fibres)} ???') 
				self._fibres[idx] = fibre
			fibre.meta = d				# CONTAINS x,y,index,factor

		# FINIS
		if show :
			self.plot_traces (mode='horizontal', show_data=True)
			plt.tight_layout ()
			plt.show ()
		# self.plot_traces (mode='amplitudes', show_data=True)
		return True

	def get_fibre (self, idxlabel) :
		""" Returns the Fibre object with the index (int) or label (str) "idxlabel". """
		if self._fibres is None :
			return None
		for idx,fibre in self._fibres.items() :
			if fibre is None :
				return None
			else :
				label = fibre.label
				if isinstance(idxlabel,int) and idx == idxlabel :
					return fibre
				elif isinstance(idxlabel,str) and label == idxlabel :
					return fibre
		return None

	def _get_trace (self, vslice,spectra,ilast,yoff) :
		# GET VERTICAL SLICE INFO
		xavg  = vslice['xavg']
		yavgs = vslice['yavgs']
		if 'yfit' in vslice :
			yavgs = vslice['yfit']
		facts = vslice['sigma_factors']
		amps  = vslice['amps']

		# GET NUMBER OF IDENTIFIED SPECTRA
		ns    = len(spectra)

		# PREPARE IDENTIFICATION INFO FOR SLICES
		ni = len(yavgs)
		ids     = ni*[9999]		# FINAL 0 <= IDS < ns
		diffs   = ni*[0]		# CURRENT SMALLEST DIFFERNCE OF CURRENT ID

		# ADD IDENTIFICATION INFO TO VERTICAL SLICE
		vslice['ids']   = ids
		vslice['diffs'] = diffs

		# FOR EVERY IDENTIFIED SPECTRUM j
		for j in range(ns) :
			d = spectra[j]
			yexpect = d['y'][ilast]+yoff[j]	# EXPECTED POSITION BASED ON PREVIOUS IDS

			# FIND THE CLOSEST NEIGHBORING MAXIMUM IN yavgs WITH INDEX k
			k = np.argmin(np.abs(yavgs-yexpect))
			yclose = yavgs[k]
			dy = yclose-yexpect

			# CLOSE ENOUGH?
			if dy > self.dy_min and dy < self.dy_max : 
				jj = ids[k]		# CURRENT 0 <= ID < ns
				if jj == 9999 or np.abs(diffs[k]) > np.abs(dy) : # NEED TO REPLACE ID
					if jj != 9999 :
						ids[k] = 9999		# FREE UP THE OLD ID
					ids[k]     = j			# RESET THE NEW ID
					diffs[k]   = dy

					yoff[j]  = int(dy)
					d['x'].append(xavg)
					d['y'].append(yclose)
					d['sigma_factors'].append(facts[k])
					d['amplitudes'].append(amps[k])

		# ANY REMAINING MAXIMA?
		mask = np.where(ids == 9999)[0]
		if len(mask) > 0 :
			for i in mask :
				sp[n1] = {'x':[xavg],'y':[yavgs[i]],'index':ns}
				ids2[i] = ns
				yoff.append(0)
				ns += 1
		return yoff

	def _fit (self, x,y,func) :
		"""
		Fits a polynomial to a set of x,y data.
		"""
		p,cov = curve_fit (func,x,y)
		return p

	def fit_traces (self, show=False) -> bool :
		"""
		Fits a polynomial to the horizontal traces.
		"""
		nf = len(self._fibres)
		nt = 0
		showing = show

		# FOR EACH POTENTIALLY IDENTIFIED SPECTRUM
		for idx in range(1,nf+1) :
			fibre = self.get_fibre (idx)
			d = fibre.meta
			if d is not None :
				x = d['x']
				y = d['y']
				a = d['amplitudes']
				n = len(x)

				# FIT POSITIONS
				if   n > 5 and self.trace_order >= 5 :
					fc = polynomial_functions[5]
				elif n > 4 and self.trace_order >= 4 :
					fc = polynomial_functions[4]
				elif n > 3 and self.trace_order >= 3 :
					fc = polynomial_functions[3]
				elif n > 2 and self.trace_order >= 2 :
					fc = polynomial_functions[2]
				elif n > 1 and self.trace_order >= 1 :
					fc = polynomial_functions[1]
				else :
					fc = polynomial_functions[0]
				pc = self._fit (x,y,fc)
				d['trace_func'] = fc
				d['trace_coef'] = pc
				fibre.trace_coef = pc

				# FIT AMPLITUDES
				if self.ampl_model == 'gaussians' :
					mg = MultiG ()
					mg.analyze (x,a,plot=False)
					pa = mg.parameters()
					fa = multiple_gauss_function

				else :	# self.ampl_model == 'polynomial'
					if n > 5 and self.ampl_order >= 5 :
						fa = polynomial_functions[5]
					elif n > 4 and self.ampl_order >= 4 :
						fa = polynomial_functions[4]
					elif n > 3 and self.ampl_order >= 3 :
						fa = polynomial_functions[3]
					elif n > 2 and self.ampl_order >= 2 :
						fa = polynomial_functions[2]
					elif n > 1 and self.ampl_order >= 1 :
						fa = polynomial_functions[1]
					else :
						fa = polynomial_functions[0]
					pa = self._fit (x,a,fa)

				afit = fa(x,*pa)
				d['ampl_func'] = fa
				d['ampl_coef'] = pa
				d['ampl_model'] = self.ampl_model
				fibre.ampl_coef = pa
				fibre.ampl_model = self.ampl_model

				if showing :
					xx = np.linspace(x[0],x[-1],500)
					fig = plt.figure ()
					plt.style.use ('ggplot')
					plt.xlabel ('x [pix]')
					plt.ylabel ('amplitude')
					if self.ampl_model == 'gaussians' :
						mg.plot_components (xx)
					plt.plot (x,a,'o',label='amp')
					plt.plot (x,a-afit,'o',label='residuals')
					plt.plot (xx,fa(xx,*pa),'-',label='{0} fit'.format(self.ampl_model))
					plt.plot ([x[0],x[-1]],[0,0],'--',label=None)
					plt.legend (bbox_to_anchor=(1.02,1), loc='upper left', ncol=1, fontsize='x-small')
					plt.title ('trace #{0}'.format(idx))
					plt.tight_layout ()
					reslt = show_with_menu (fig,['no more amplitude plots','save to file','ABORT'])
					if reslt == 'no more amplitude plots' :
						showing = False
						details = False
					elif reslt == 'save to file' :
						with open('trace_amplitudes.csv','w') as stream :
							stream.write ('x, amp, fit\n')
							for i in range(len(x)) :
								stream.write (f'{x[i]:.3f}, {a[i]:.3f}, {afit[i]:.3f}\n')
					elif reslt == 'ABORT' :
						return False

				nt += 1
			else :
				logging.warning ('no metadata for fibre #{0}'.format(idx))

		self.number_traces = nt
		logging.info ('fit {0} traces'.format(nt))

		# NOTE TRACE FIT COEFFICIENTS FOR LATER
		self.save_coefficients ()

		# FINIS
		return True

	def save_coefficients (self, pathname=None) :
		"""
		Store the trace and amplitude coefficients either in a YAML file (pathname) and/or in
		the internal HDU's FITS header.
		"""
		if self.hdu is not None :
			hdr = self.hdu.header
		else :
			hdr = None

		tbylabel = {}
		tbyindex = {}
		abylabel = {}
		abyindex = {}
		mt = 0
		ma = 0
		nf = self.number_traces
		n = 0

		for idx in range(1,nf+1) :
			fibre = self.get_fibre (idx)
			pt = fibre.trace_coef
			mt = len(pt)
			pa = fibre.ampl_coef
			ma = len(pa)
			if pt is not None and pa is not None :
				if hdr is not None :
					fibre.update_header (hdr,mode='multiple')
				n += 1

				# STORE COEFFICIENTS FOR YAML
				if pathname is not None :
					tbyn = len(pt)*[0]
					tbyi = len(pt)*[0]
					for l in range(mt) :
						tbyn[l] = float(pt[l])
						tbyi[l] = float(pt[l])
					tbylabel[fibre.label] = tbyn
					tbyindex[idx]         = tbyi

					abyn = len(pa)*[0]
					abyi = len(pa)*[0]
					for l in range(ma) :
						abyn[l] = float(pa[l])
						abyi[l] = float(pa[l])
					abylabel[fibre.label] = abyn
					abyindex[idx]         = abyi

			else :
				logging.error ('cannot save_trace coefficients without coefficients for {0}!'.format(idx))

		if n != nf :
			logging.info ('only {0} of {1} traces have pos. and ampl. fits!'.format(n,nf))

		# SAVE sigma COEFFICIENTS IN HEADER
		p = self._sigma_fits
		if hdr is not None :
			ok = True
			for keyw in ['sigma_order','sigma_fit0','sigma_fit1','sigma_fit2','sigma_fit3'] :
				if keyw not in self.keywords :
					ok = False
			if (p is not None) and ok :
				keyw,comment = self.keywords['sigma_order']
				hdr[keyw] = self.sigma_order,comment
				for i in range(4) :
					keyw,comment = self.keywords['sigma_fit{0}'.format(i)]
					if self.sigma_order >= i :
						hdr[keyw] = p[i],comment
					else :
						hdr[keyw] = 0.,comment

			# SAVE TRACE INFO
			if 'trace_order' in self.keywords :
				keyw,comment = self.keywords['trace_order']
				hdr[keyw] = mt-1,comment
			if 'trace_format' in self.keywords :
				keyw,comment = self.keywords['trace_format']
				hdr[keyw] = self.trace_format,comment
			if 'ampl_order' in self.keywords :
				keyw,comment = self.keywords['ampl_order']
				hdr[keyw] = ma-1,comment
			if 'ampl_format' in self.keywords :
				keyw,comment = self.keywords['ampl_format']
				hdr[keyw] = self.trace_format,comment
			if 'ampl_model' in self.keywords :
				keyw,comment = self.keywords['ampl_model']
				hdr[keyw] = self.ampl_model,comment
			else :
				logging.warning (f'"ampl_model" not in {str(self.keywords)}???')
			if 'number_slices' in self.keywords :
				keyw,comment = self.keywords['number_slices']
				hdr[keyw] = self.number_traces,comment
			if 'number_traces' in self.keywords :
				keyw,comment = self.keywords['number_traces']
				hdr[keyw] = n,comment

		# SAVE YAML DICTIONARY WITH trace CONFIGURATION
		if pathname is not None :
			info = {}
			info['datetime']       = UTC_now()
			info['number_traces']  = nf
			info['positions'] = {'by_index':tbyindex, 'by_label':tbylabel,
							'trace_order':mt-1, 'trace_format':self.trace_format}
			info['amplitudes'] = {'by_index':abyindex, 'by_label':abylabel,
							'ampl_order':ma-1, 'ampl_format':self.ampl_format, 'ampl_model':self.ampl_model}
			if p is not None :
				info['sigma_fits'] = [float(x) for x in p]		# list(p) DOES NOT WORK??????????
			cfg = {'trace':info}
			try :
				with open (pathname,'w') as stream :
					yaml.dump (cfg,stream, default_flow_style=False, default_style='')
			except Exception as e :
				logging.error ('cannot store YAML configuration:'+str(e))

	def number_of_traces (self) -> int :
		"""
		Number of already determined spectral traces.
		"""
		if self._fibres is None :
			return -1
		else :
			n = 0
			for idx,fibre in self._fibres.items() :
				if fibre is not None and fibre.trace_coef is not None and fibre.ampl_coef is not None :
					n += 1
			self.number_traces = n
			return n

	def get_trace_position_model (self, idx) :
		"""
		Returns the polynomial coefficients and function for the positions.
		"""
		fibre = self.get_fibre (idx)
		if fibre is None :
			raise IndexError ('#{0} is not a valid trace index (no fibre)'.format(idx))
		pc = fibre.trace_coef
		if pc is None :
			raise IndexError ('#{0} is not a valid trace position index (no coefficients)'.format(idx))
		return pc,polynomial_functions[len(pc)-1]

	def get_trace_amplitude_model (self, idx) :
		"""
		Returns the polynomial coefficients and function for the amplitudes.
		"""
		fibre = self.get_fibre (idx)
		if fibre is None :
			raise IndexError ('#{0} is not a valid trace index (no fibre)'.format(idx))
		pa = fibre.ampl_coef
		if pa is None :
			raise IndexError ('#{0} is not a valid trace amplitude index (no coefficients)'.format(idx))
		return pa,polynomial_functions[len(pa)-1]

	def get_trace_position_coefficients (self, idx) :
		"""
		Gets the polynomial coefficents of the spatial trace for the idx-th spectrum.
		"""
		fibre = self.get_fibre (idx)
		if fibre is None :
			logging.error ('no fibre available for #{0}'.format(idx))
			return None
		return fibre.trace_coef

	def get_trace_amplitude_coefficients (self, idx) :
		"""
		Gets the polynomial coefficents of the spatial trace for the idx-th spectrum.
		"""
		fibre = self.get_fibre (idx)
		if fibre is None :
			logging.error ('no fibre available for #{0}'.format(idx))
			return None
		return fibre.ampl_coef

	def plot_traces (self, mode='horizontal', show_data=False, kappa=1.0, title=None, width=None, *args, **kwargs) -> bool :
		"""
		Plots various results of the tracing procedures.
		"""
		if self.number_traces is None or self.number_traces <= 0 :
			logging.warning ('no traces to plot!')
			return False

		c = ['b','g','r','c','m','k']
		ny,nx = self.hdu.shape
		if show_data :
			show_hdu (self.hdu, aspect='auto', kappa=kappa, **kwargs)
			c = ['w','r']

		if mode == 'vertical' :
			if self._vert_slices is None :
				logging.error ('no vertical traces to plot!')
				return False
			for d in self._vert_slices :
				if 'yavgs' in d and 'xavg' in d :
					coords = d['yavgs']
					n = len(coords)
					x = np.zeros(n)+d['xavg']
					plt.plot (x,coords,'+')
			if title is None :
				plt.title ('vertical traces')

		elif mode == 'amplitudes' :
			if self._fibres is None :
				logging.error ('nothing to plot!')
				return False
			ic = 0
			for idx in range(1,self.number_traces+1) :
				fibre = self.get_fibre (idx)
				if fibre is None :
					logging.error ('cannot access trace #{0}'.format(idx))
					return False
				htrace = fibre.meta
				if htrace is not None :
					logging.debug ('plotting amplitude #{0} = spectrum {1}'.format(idx,fibre.label))
					if 'x' in htrace and 'amplitudes' in htrace :
						x = np.array(htrace['x'])
						a = np.array(htrace['amplitudes'])
						plt.plot (x,a,'x',color='red')	# c[ic])
					else :
						logging.debug('no x,ampl data for trace #{0}'.format(idx))
				x = np.linspace (0,nx-1,500)
				if fibre.ampl_coef is not None and fibre.ampl_model is not None :
					p = fibre.ampl_coef
					if fibre.ampl_model == 'gaussians' :
						f = multiple_gauss_function
					else :
						f = polynomial_functions[len(p)-1]
					afit = f (x,*p)
					plt.plot (x,afit,'-',color='red')	# c[ic])
				ic = (ic+1)%len(c)
			if title is None :
				plt.title ('amplitude traces')

		else :
			if self._fibres is None :
				logging.error ('nothing to plot!')
				return False
			ic = 0
			for idx in range(1,self.number_traces+1) :
				fibre = self.get_fibre (idx)
				if fibre is None :
					logging.error ('cannot access trace #{0}'.format(idx))
					return False
				htrace = fibre.meta
				if htrace is not None :
					logging.info ('plotting trace #{0} = spectrum {1}'.format(idx,fibre.label))
					if 'x' in htrace and 'y' in htrace :
						x = np.array(htrace['x'])
						y = np.array(htrace['y'])
						plt.plot (x,y,'x',color='red')	# c[ic])
					else :
						logging.warning ('no x,y data for trace #{0}'.format(idx))
				x = np.linspace (0,nx-1,500)
				if fibre.trace_coef is not None :
					p = fibre.trace_coef
					f = polynomial_functions[len(p)-1]
					yfit = f (x,*p)
					if width is None :
						plt.plot (x,yfit,'-',color='red')	# c[ic])
					else :
						plt.plot (x,yfit+width//2,'--',color='red')	# c[ic])
						plt.plot (x,yfit-width//2,'--',color='red')	# c[ic])
					if fibre.label is not None :
						plt.text (x[-1],yfit[-1],' '+fibre.label,color='red',va='center',fontsize='small')
				ic = (ic+1)%len(c)
			if title is None :
				plt.title ('horizontal traces')

		plt.xlabel ('x [pix]')
		plt.ylabel ('y [pix]')
		if title is not None :
			plt.title (title)
		return True

	def find_spectra (self, show=True, details=False) -> bool :
		"""
		Identifies the spectra from their traces.
		"""
		logging.info (f'ampl_order={self.ampl_order}')
		if not self.find_maxima (show=details) :
			logging.critical ('could not find maxima')
			return False
		elif not self.trace_spectra (show=details) :
			logging.critical ('could not trace spectra')
			return False
		elif not self.fit_profiles(show=show, details=details) :
			logging.warning ('could not fit all vertical profiles')
		return True

def main () :
	from pyFU.utils import parse_arguments, initialize_logging

	# ---- GET DEFAULTS AND PARSE COMMAND LINE
	arguments = {
		'ampl_model':{'path':'trace:','default':'polynomial',
			'flg':'-A','type':str,'help':'x model for traces ("polynomial"|"gaussians")'},
		'ampl_order':{'path':'trace:','default':5,
			'flg':'-a','type':int,'help':'order of polynomial fit to spectra amplitudes (default 5)'},
		'bgk_factor':{'path':'trace:','default':1,
			'flg':'-b','type':float,'help':'background factor'},
		'trace_bias':{'path':'trace:','default':0.,
			'flg':'-B','type':float,'help':'trace background bias level'},
		'details':{'path':None,'default':False,
			'flg':'-d','type':bool,  'help':'show details'},
		'slices':{'path':None,'default':None,
			'flg':'-E','type':str,'help':'name of external CSV file with list of slice positions (xavg,xleft,dx)'},
		'generic':{'path':None,'default':None,
			'flg':'-G','type':str,'help':'YAML file for generic trace configuration info'},
		'sigma':{'path':'trace:','default':3,
			'flg':'-g','type':float,'help':'Gaussian width of spectra'},
		'infile': {'path':'trace:','default':None,
			'flg':'-i','type':str,'help':'FITS file name (default ./spectra/test.fits)'},
		'ignore': {'path':None,'default':False,
			'flg':'-I','type':bool,'help':'ignore trace info in FITS header (default False)'},
		'mid_slice':{'path':'trace:','default':None,'dshow':'middle',
			'flg':'-m','type':int,'help':'middle slice used to find spectra'},
	    'number_slices':{'path':'trace:','default':30,
			'flg':'-N','type':int,'help':'number of vertical slices'},
	    'number_fibres':{'path':'trace:','default':None,
			'flg':'-n','type':int,'help':'number of IFU fibres'},
	    'plot':{'path':None,'default':None,
			'flg':'-p','type':str,'help':'matplotlib backend (e.g. MacOSX|TkAgg,..) (default None)'},
		'sigma_order':{'path':'trace:','default':3,
			'flg':'-o','type':int,'help':'polynomial order of fit to width variations'},
		'trace_order':{'path':'trace:','default':5,
			'flg':'-O','type':int,'help':'max. order of polynomial fit to spectra positions'},
	    'window_profile':{'path':'trace:','default':9,
			'flg':'-P','type':int,'help':'vertical profile window'},
		'dy_min':{'path':'trace:','default':-4,
			'flg':'-s','type':int,'help':'minimum expected spacing of spectra'},
		'dy_max':{'path':'trace:','default':4,
			'flg':'-S','type':int,'help':'maximum expected spacing of spectra'},
	    'number_traces':{'path':'trace:','default':None,
			'flg':'-T','type':int,'help':'number of traced spectra'},
		'window_centroid':{'path':'trace:','default':5,
			'flg':'-W','type':int,'help':'width of centroid window'},
		'window_max':{'path':'trace:','default':5,
			'flg':'-w','type':int,'help':'width of maximum search window'},
		'spacing':{'path':'trace:','default':7,
			'flg':'-x','type':float,'help':'vertical spacing of spectra'},
		'save':{'path':'trace:','default':None,
			'flg':'-Y','type':str,'help':'pathname of output YAML file for trace coefficents'},
		'yaml':{'path':None,'default':None,
			'flg':'-y','type':str,'help':'global YAML configuration file for parameters'}
		}
	args,cfg = parse_arguments (arguments)
	info = cfg['trace']

	# ---- LOGGING
	initialize_logging (config=cfg)
	logging.info ('*************************** trace ******************************')

	# ---- OUTPUT GENERIC CONFIGURATION FILE?
	if args.generic is not None :
		logging.info ('Appending generic trace configuration info to'+args.generic)
		with open (args.generic,'a') as stream :
			yaml.dump ({'trace':info}, stream)
		sys.exit(0)

	# ---- GET INPUT DATA
	if 'infile' not in info or info['infile'] is None :
		logging.critical ('no input filename!')	# NEITHER CONFIG NOR ARGS
		sys.exit(1)
	logging.debug ('infile: {0}'.format(info['infile']))
	hdus = fits.open (info['infile'],mode='update')
	hdu = hdus[0]

	# ---- GET TRACER
	tracer = SpectrumTracer (hdu, config=cfg, ignore=args.ignore)

	# ---- GET EXTERNAL CSV FILE WITH VERTICAL SLICE POSITIONS
	if args.slices is not None :
		tab = Table.read (args.slices, format='ascii.csv')
		logging.debug (f'table with vertical slice positions:\n{str(tab)}')
		tracer.use_external_slices (tab)
		"""
		except :
			print ('cannot load external slices from CSV file ',args.slices)
			sys.exit(1)
		"""

	# ---- FIND SPECTRA
	tracer.find_spectra (show=args.plot, details=args.details)

	# ---- FIT TRACES
	tracer.fit_traces (show=args.plot)

	# ---- UPDATE HDU HEADER
	hdus.flush()

	# ---- SAVE TRACE COEFFICIENTS IN A YAML FILE
	if 'save' in info and info['save'] is not None :
		logging.info ('Saving trace coefficients..')
		tracer.save_coefficients (info['save'])

	logging.info ('****************************************************************')

if __name__ == '__main__' :
	main ()

