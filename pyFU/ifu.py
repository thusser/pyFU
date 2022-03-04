# pyfu/ifu.py 

"""
Objects for managing IFU fibres and their spectra.
"""

import numpy as np
import sys
import logging

from astropy.io    import fits
from astropy.table import Table, Column
from astropy.wcs   import WCS

from matplotlib          import pyplot as plt, cm as colourmaps
from matplotlib.colors   import Normalize
from matplotlib.colorbar import ColorbarBase

from pyFU.defaults import pyFU_default_formats, pyFU_default_keywords, pyFU_logging_level, pyFU_logging_format
from pyFU.meta     import header2config
from pyFU.utils    import hist_integral, merge_dictionaries, polynomial_functions, multiple_gauss_function

# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)


class Fibre (object) :
	""" 
	Object containing the information about a single IFU fibre and its trace
	- label, focal position, diameters, widths, trace coefficients  -
	extractable from a FITS header or from a configuration dictionary.
	The "meta" attribute is for the storage of misc data by client objects/methods.
	"""
	def __init__ (self, idx=None, header=None, config=None, keywords=None, formats=None) :
		"""
		If no fibre index 1 <= idx <= number_fibres is given,
		then the header is assumed to refer to a single fibre, as is the
		case for extracted spectral tables..
		"""
		self.index = None					# INDEX ON SPECTRAL IMAGE >= 1
		self.label = None					# LABEL
		self.pos  = None					# FOCAL PLANE POSITION [x,y]
		self.inner_diameter = None
		self.outer_diameter = None
		self.trace_coef = None				# COEFFICIENTS FOR Y(X)   IN SPECTRAL IMAGE
		self.ampl_coef = None				# COEFFICIENTS FOR A(X)   IN SPECTRAL IMAGE
		self.ampl_model = 'polynomial'	    # FUNCTION USED FOR A(X)  IN SPECTRAL IMAGE
		self.sigma = None					# GAUSSIAN VERTICAL WIDTH IN SPECTRAL IMAGE
		self.spacing = None					# VERTICAL WIDTH          IN SPECTRAL IMAGE
		self.meta = None

		self.keywords = keywords
		if keywords is None :
			self.keywords = dict(pyFU_default_keywords)

		self.formats = formats
		if formats is None :
			self.formats = dict(pyFU_default_formats)

		# ADOPT GIVEN INDEX
		self.index = idx

		# COMBINE INFO FROM GENERIC CONFIGURATION AND SPECIFIC HEADER
		if (header is not None) and (config is not None) :
			cfg = dict(config)
			hcfg = header2config (header, keywords=self.keywords, formats=self.formats)
			merge_dictionaries (cfg,hcfg)
			self._init_from_config (cfg)
		# OR USE HEADER 
		elif header is not None :
			hcfg = header2config (header, keywords=self.keywords, formats=self.formats)
			self._init_from_config (hcfg)
			# self._init_from_header (idx=idx,header=header)
		# OR USE GENERIC CONFIGURATION
		elif config is not None :
			self._init_from_config (config)
		else :
			logging.error ('cannot initialize fibre without header or config')

	def __str__ (self) :
		s = '#{0}: label={1}, pos={2}, '.format(self.index,self.label, str(self.pos))

		if self.trace_coef is None :
			s += 'no trace, '
		else :
			s += 'len(trace_coef)={0}, '.format(len(self.trace_coef))

		if self.ampl_coef is None :
			s += 'no ampl, '
		else :
			s += 'len(ampl)={0}, '.format(len(self.ampl_coef))

		if self.ampl_model is None :
			s += 'polyn. ampl, '
		else :
			s += 'mgauss ampl'

		if self.inner_diameter is None or self.outer_diameter is None :
			s += 'diam={0},{1}, '.format(self.inner_diameter,self.outer_diameter)
		else :
			s += 'diam={0:.2f},{1:.2f}, '.format(self.inner_diameter,self.outer_diameter)

		if self.spacing is None :
			s += 'spacing=None, '
		else :
			s += 'spacing={0:.2f}, '.format(self.spacing)

		if self.sigma is None :
			s += 'sigma=None, '
		else :
			s += 'sigma={0:.2f}, '.format(self.sigma)

		s += 'meta={0}'.format(str(self.meta))
		return s

	def _init_from_config (self, config) :
		"""
		Get info from a pyFU configuration file.
		"""
		if config is None or len(config) == 0 :
			logging.error ('cannot initialize fibre from a null/empty config dictionary!')
			return
		info = config
		labels = None

		# GET IFU CONFIGURATIONS...
		if 'ifu' in config :
			info = config['ifu']

		# ... GET INDEX FROM 1 TO number_fibres
		if 'index' in config :
			idx = config['index']
			if self.index is not None and self.index != idx :
				raise ValueError ('internal index does not match index!')
			self.index = idx
		if self.index is None :
			logging.error ('cannot initialize fibre from config without index!')
			return
		logging.debug (f'fibre index={self.index}')

		# ... GET LABEL
		self.label = f'f#{self.index}'		# DEFAULT LABEL
		if 'label' in config and config['label'] is not None :
			self.label = config['label']
		elif 'slit_labels' not in info or info['slit_labels'] is None :
			logging.debug ('label and list of labels not available - using default!')
		else :
			labels = info['slit_labels']	# DICTIONARY WITH NON-ZERO INTEGER KEYS ?
			if isinstance(labels,list) :	# CONVERT LIST TO DICTIONARY
				labels = {idx+1:labels[idx] for idx in range(len(labels))}
			idx0 = 0						# STARTING POINT SHIFT IN LABEL ARRAY?
			if ('slit_start' in info) and (info['slit_start'] is not None) :
				idx0 = info['slit_start']-1

			if labels is not None :
				if len(labels) == 1 :		# JUST ONE LABEL FOR ONE FIBRE
					self.label = labels[1]
				else :
					ii = self.index+idx0
					if ii <= 0 or ii > len(labels) :
						logging.warning (f'fibre label {ii} does not match index {self.index} - using default')
					else :
						# print ('ii=',ii,', labels=',labels)
						self.label = labels[ii]

		logging.debug (f'labels:{labels}')
		logging.debug ('fibre label={0}'.format(self.label))

		# ... GET POSITION
		self.pos = None
		if 'xpos' in config and 'ypos' in config :
			self.pos = [config['xpos'],config['ypos']]
		elif 'focal_positions' not in info :
			logging.debug ('dict of fibre focal positions not available - using index!')
		else :
			positions = info['focal_positions']
			if self.label in positions :
				self.pos = positions[self.label]

		# ... GET DIAMETERS
		if 'inner_diameter' in info :
			self.inner_diameter = info['inner_diameter']	# GENERIC DIAMETER
		if 'inner_diameter' in config :
			self.inner_diameter = config['inner_diameter']	# SPECIFIC DIAMETER
		if 'inner_diameters' in info :
			d = info['inner_diameters']
			if self.label in d :
				self.inner_diameter = d[self.label]

		if 'outer_diameter' in info :
			self.outer_diameter = info['outer_diameter']	# GENERIC DIAMETER
		if 'outer_diameter' in config:
			self.outer_diameter = config['outer_diameter']	# SPECIFIC DIAMETER
		if 'outer_diameters' in info :
			d = info['outer_diameters']
			if self.label in d :
				self.outer_diameter = d[self.label]

		# GET TRACE CONFIGURATION ...
		info = config
		if 'trace' in config :
			info = config['trace']

		# ... SPACING
		if 'spacing' in info :
			self.spacing = info['spacing']
		if 'spacings' in info :
			d = info['spacings']
			if self.label in d :
				self.spacing = d[self.label]

		# ... SIGMAS
		if 'sigma' in info :
			self.sigma = info['sigma']
		if 'sigma' in config :
			self.sigma = config['sigma']
		if 'sigmas' in info :
			d = info['sigmas']
			if self.label in d :
				self.sigma = d[self.label]

		# ... TRACE COEFFICIENTS
		if 'positions' in info :
			fit = info['positions']
			if 'by_index' in fit :
				if self.index in fit['by_index'] :
					self.trace_coef = fit['by_index'][self.index]
			elif 'by_label' in fit :
				if self.label in fit['by_label'] :
					self.trace_coef = fit['by_label'][self.label]
			else :
				logging.warning (str(fit))
				logging.warning ('no by_index or by_label in fit:positions dictionary')
		else :
			logging.info ('no config["trace"]["fit"]["positions"]')

		# ... AMPLITUDE COEFFICIENTS
		if 'amplitudes' in info :
			fit = info['amplitudes']
			if 'by_index' in fit :
				if self.index in fit['by_index'] :
					self.ampl_coef = fit['by_index'][self.index]
			elif 'by_label' in fit :
				if self.label in fit['by_label'] :
					self.ampl_coef = fit['by_label'][self.label]
			else :
				logging.warning (str(fit))
				logging.warning ('no by_index or by_label in fit:amplitudes dictionary')
		else :
			logging.info ('no config["trace"]["fit"]["amplitudes"]')

		# ... AMPLITUDE FUNCTION
		if 'ampl_model' in info :
			self.ampl_model = info['ampl_model']
		else :
			logging.warning ('no amplitude model in configuration: assuming "polynomial')
			self.ampl_model = 'polynomial'

	def _init_from_header (self, header, idx=None) :
		"""
		Get info from a FITS/Table header.  If the header has no particular info
		(e.g. which index is being referred to), such is left un-touched. 
		"""
		if header is None :
			logging.error ('cannot initialize fibre from a null header!')
			return
		if idx is not None :
			self.index = idx
		cfg = header2config (header, keywords=self.keywords, formats=self.formats)
		self._init_from_config (cfg)

	def update_header (self, header, mode='single') -> bool :
		"""
		Add the Fibre's parameters to a FITS header for a single fibre.
		if the mode is not 'single', then the multiple-fibre syntax is used.
		"""
		if self.index is None :
			logging.warning ('cannot update FITS header without a fibre index')
			return False

		# SAVE TRACE COEFFICIENTS
		if self.trace_coef is not None :

			# GET FORMAT
			tfmt = None
			if 'trace_format' in self.formats :		# GET FROM FORMATS
				tfmt = self.formats['trace_format']
				if 'trace_format' in self.keywords :
					keywkey,comment = self.keywords['trace_format']
				else :
					keywkey.comment = pyFU_default_keywords['trace_format']
				header[keywkey] = (tfmt,comment)

			elif 'trace_format' in self.keywords :	# GET FROM HEADER
				keywkey = self.keywords['trace_format'][0]
				if keywkey in header :
					tfmt = header[keywkey]

			if tfmt is None :
				logging.error ('fibre #{0} does not have a trace coefficent format'.format(self.index))
				return False

			# SAVE COEFFICIENTS
			p = self.trace_coef
			np = len(p)
			for k in range(10) :
				keyw = tfmt.format(self.index,k)
				if 'trace_coef' in self.keywords :
					comment = self.keywords['trace_coef'][1]
				else :
					comment = pyFU_default_keywords['trace_coef'][1]
				if k < np :
					header[keyw] = (p[k],comment)
				elif keyw in header :	# LEFT OVER FROM BEFORE?  NEED TO REMOVE
					header.remove (keyw)

			# NOTE ORDER
			if 'trace_order' in self.keywords :
				keywkey,comment = self.keywords['trace_order']
				header[keywkey] = (np-1,comment)

		# SAVE AMPLITUDE COEFFICIENTS
		if self.ampl_coef is not None :

			# GET FORMAT
			afmt = None
			if 'ampl_format' in self.formats :
				afmt = self.formats['ampl_format']
				if 'ampl_format' in self.keywords :
					keywkey,comment = self.keywords['ampl_format']
				else :
					keywkey.comment = pyFU_default_keywords['ampl_format']
				header[keywkey] = (afmt,comment)

			elif 'ampl_format' in self.keywords :
				keywkey = self.keywords['ampl_format'][0]
				if keywkey in header :
					afmt = header[keywkey]
			if afmt is None :
				logging.error ('fibre #{0} does not have an amplitude coefficent format'.format(self.index))
				return False

			# SAVE COEFFICIENTS
			p = self.ampl_coef
			np = len(p)
			for k in range(10) :
				keyw = afmt.format(self.index,k)
				if 'ampl_coef' in self.keywords :
					comment = self.keywords['ampl_coef'][1]
				else :
					comment = pyFU_default_keywords['ampl_coef'][1]
				if k < np :
					header[keyw] = (p[k],comment)
				elif keyw in header :	# LEFT OVER FROM BEFORE?  NEED TO REMOVE
					header.remove (keyw)

			# NOTE ORDER
			if 'ampl_order' in self.keywords :
				keywkey,comment = self.keywords['ampl_order']
				header[keywkey] = (np-1,comment)

			# NOTE FUNCTION
			if 'ampl_model' in self.keywords :
				keywkey,comment = self.keywords['ampl_model']
				header[keywkey] = (self.ampl_model,comment)

		# SOME COMPLEX METADATA TO SAVE
		xpos,ypos = None,None
		if self.pos is not None :
			xpos,ypos = self.pos

		# SAVE INDEX FOR SINGLE FIBRE FORMAT
		if mode == 'single' :
			things = [ \
				(self.index,         'index') \
				]
			for thing,key in things :
				if thing is not None :
					if key in self.keywords :
						keyw,comment = self.keywords[key]
					else :
						keyw,comment = pyFU_default_keywords[key]
					header[keyw] = thing,comment

		things = [ \
			(self.index,         'index_format',         'index'), \
			(self.label,         'label_format',         'label'), \
			(xpos,               'xpos_format',          'xpos'), \
			(ypos,               'ypos_format',          'ypos'), \
			(self.inner_diameter,'inner_diameter_format','inner_diameter'), \
			(self.outer_diameter,'outer_diameter_format','outer_diameter'), \
			(self.spacing,       'spacing_format',       'spacing'), \
			(self.sigma,         'sigma_format',         'sigma') \
			]
		for thing,fmtkey,key in things :
			if thing is not None :
				# GET THE FORMAT
				if fmtkey not in self.formats :
					logging.warning ('fibre #{0} has no header format for {1}'.format(self.index,fmtkey))
					fmt = pyFU_default_formats[fmtkey]
				else :
					fmt = self.formats[fmtkey]

				# PRODUCE THE FITS KEYWORD
				keyw = fmt.format(self.index)

				# GET THE COMMENT
				if key in self.keywords :
					comment = self.keywords[key][1]
				else :
					comment = pyFU_default_keywords[key][1]

				# SAVE THING
				header[keyw] = (thing,comment)

				# ALSO SAVE THE FORMAT ITSELF
				if fmtkey in self.keywords and self.keywords[fmtkey][0] not in header :
					key,comment =  self.keywords[fmtkey]
					header[key] = fmt,comment

	def get_trace_position_model (self) :
		"""
		Returns the polynomial coefficients and function for the positions.
		Equivalent to SpectrumTracer.get_trace_position_model (idx).
		"""
		if self.trace_coef is None :
			raise IndexError ('no trace model for fibre #{0}/{1}'.format(self.index,self.label))
		pc = np.array(self.trace_coef)
		return pc,polynomial_functions[len(pc)-1]

	def get_trace_amplitude_model (self) :
		"""
		Returns the polynomial coefficients and function for the amplitudes.
		Equivalent to SpectrumTracer.get_trace_amplitude_model (idx).
		"""
		if self.ampl_coef is None :
			raise IndexError ('no amplitude coefficients for fibre #{0}/{1}'.format(self.index,self.label))
		if self.ampl_model is None :
			raise IndexError ('no amplitude model for fibre #{0}/{1}'.format(self.index,self.label))
		pa = np.array(self.ampl_coef)
		if self.ampl_model == 'polynomial' :
			return pa,polynomial_functions[len(pa)-1]
		else :
			return pa,multiple_gauss_function

def hexpos (nfibres,diam) :
	"""
	Returns a list of [x,y] positions for a classic packed hex IFU configuration.
	"""
	positions = [[np.nan,np.nan] for i in range(nfibres)]

	# FIND HEX SIDE LENGTH
	nhex = 1
	lhex = 1
	while nhex < nfibres :
		lhex += 1
		nhex = 3*lhex**2-3*lhex+1
	if nhex != nfibres:
		lhex -= 1
	nhex = 3*lhex**2-3*lhex+1
	nextra = nfibres-nhex
	n = 0
	khex = 2*lhex-1	# NUMBER OF FIBRES IN THE CENTRAL ROW
	xhex = (-khex//2)*diam
	for i in range(khex) :	# CENTRAL ROW
		x = xhex+diam*i
		positions[n] = [int(x*100)/100,0.]
		n += 1
	
	dx = 0.5*diam
	dy = diam*np.sqrt(3./4.)
	for i in range(1,lhex,1) :		# FOR ALL ROWS PAIRS i
		khex -= 1					# EACH ROW HAS 1 LESS THAN THE PREVIOUS
		xhex += dx
		for j in range(khex) :		# FOR ALL FIBRES j IN ROWS i
			x = xhex+diam*j
			y = dy*i
			positions[n]   = [int(x*100)/100, int(y*100)/100]
			positions[n+1] = [int(x*100)/100,-int(y*100)/100]
			n += 2

	return positions

def get_fibres (config=None, header=None, keywords=pyFU_default_keywords, formats=pyFU_default_formats) :
	"""
	Get a list of pyFU fibre objects derived from a pyFU configuration or a full set
	of parameters stored in the FITS header of a traced spectral image.
	"""
	fibres = []
	n = 0

	# PARSE HEADER FOR UPDATED CONFIGURATION
	cfg = {}
	if config is not None :
		merge_dictionaries (cfg,config)
	if header is not None :
		merge_dictionaries (cfg, header2config(header, keywords=keywords, formats=formats))

	# GET NUMBER OF FIBRES
	if 'trace' in cfg and 'number_traces' in cfg['trace'] :
		n = cfg['trace']['number_traces']
	elif 'ifu' in cfg and 'slit_labels' in cfg['ifu'] :
		n = len(cfg['ifu']['slit_labels'])
	elif 'ifu' in cfg and 'number_fibres' in cfg['ifu'] :
		n = cfg['ifu']['number_fibres']
	if n is None or n == 0 :
		logging.error ('cannot determine number of fibres')
		return fibres
	logging.info (f'parsing {n} fibres...')

	# GET FIBRES USING RE-CONSTRUCTED CONFIGURATION DICTIONARY ONLY
	# SINCE THE FITS HEADER SHOULD HAVE BEEN MERGED INTO IT
	for idx in range(1,n+1) :
		fibre = Fibre (idx=idx, config=cfg, keywords=keywords, formats=formats)
		fibres.append(fibre)
		logging.info ('got fibre '+str(fibre))
	logging.info ('retrieved {0} fibres from config'.format(n))
	return fibres

def main () :
	logging.critical ('ifu does not have a main method!')

