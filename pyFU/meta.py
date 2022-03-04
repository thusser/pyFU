# pyFU/meta.py

"""
The present metadata structure (e.g. YAML files, configuration dictionaries) for the
most important entries is :

config :

	extract :
		errcol : ()				# NAME OF FLUX ERROR COLUMN (DEFAULT "err_flux")
		fluxcol : ()			# NAME OF FLUX COLUMN (DEFAULT "flux")
		gain : ()				# GAIN IN e-/ADU (DEFAULT 1.0)
		idxcol : ()				# NAME OF INDEX COLUMN (DEFAULT "index")
		infiles : []			# LIST OF INPUT FILES,  e.g. "/mydir/in*.fits"
		labcol : ()				# NAME OF FIBRE LABEL COLUMN (DEFAULT "label")
		outfiles : []			# LIST OF OUTPUT FILES, e.g. "/mydir/out*.fits"
		pixcol : ()				# NAME OF PIXEL COLUMN (DEFAULT "pixel")
		ron : ()				# RON IN e- (DEFAULT 0.0)

	ifu :
		focal_positions : {}
		inner_diameter : ()		# GENERIC VALUE
		outer_diameter : ()		# GENERIC VALUE
		inner_diameters : {}	# SPECIAL LABEL-KEYED VALUES
		outer_diameters : {}	# SPECIAL LABEL-KEYED VALUES
		slit_labels : []

	image :
		fudge : ()				# FACTOR FOR CHANGING THE IMAGE PROPORTIONS (DEFAULT 1.35)
		infiles : []			# LIST OF INPUT FILES
        info : []               # FITS KEYWORDS TO DISPLAY IN IMAGES (DEFAULT 'OBJECT','DATETIME','RA','DEC','OBSERVER','INSTRUME')
		logscale : (T/F)		# USE A LOG INTENSITY SCALE
		outfiles : []			# LIST OF OUTPUT FILES
		pixels : [p1,p2]		# PIXEL INTEGRATION RANGE
		waves : [w1,w2]			# WAVELENGTH INTEGRATION RANGE

	index : ()					# USED FOR SINGLE FIBRE
	inner_diameter : ()			# USED FOR SINGLE FIBRE
	formats : {}
	label : ()					# USED FOR SINGLE FIBRE

	lamp :
		flxcol : ()				# NAME OF FLUX COLUMN
		infiles : []			# LIST OF INPUT FILES
		model : ()				# ("linear"|"quadratic"|"cubic"|"exp"|"power")
		outfiles : []			# LIST OF OUTPUT FILES
		pixcol : ()				# NAME OF PIXEL COLUMN
		wavcol : ()				# NAME OF WAVELENGTH COLUMN
		wavetable: ()			# PATH TO REFERENCE WAVELENGTH TABLE

	outer_diameter : ()			# USED FOR SINGLE FIBRE

	rebin :
		dispersion : ()			# AMOUNT OF FINAL LINEAR DISPERSION [nm/pixel]
		errcol : ()				# NAME OF FLUX ERROR COLULMN
		flxcol : ()				# NAME OF FLUX COLUMN
		infiles : []			# LIST OF INPUT FILES
		limits : [w1,w2]		# FINAL INTERPOLATED WAVELENGTH LIMITS
		npixels : ()			# NUMBER OF FINAL INTERPOLATED PIXELS
		outfiles : []			# LIST OF OUTPUT FILES
		pixcol : ()				# NAME OF PIXEL COLUMN
		reference : ()			# PATH OF REFERENCE TABLE
		wavcol : ()				# NAME OF WAVELENGTH COLUMN
		wcol : ()				# NAME OF WAVELENGTH COLUMN IN REFERENCE TABLE
		wresolution : ()		# AMOUNT OF FINAL CENTRAL WAVELENGTH RESOLUTION [wav/dwav]

	sigma : ()					# USED FOR SIGNLE FIBRE

	trace :
		ampl_format : ()
		ampl_order : ()
		ampl_model : ()		# "polynomial" OR "gaussians"
		amplitudes :
			by_label : {}		# AMPLITUDE COEFFICIENTS
			by_index : {}		# IBID
		positions :
			by_label : {}		# POSITION COEFFICIENTS
			by_index : {}		# IBID
		trace_format : ()
		trace_order : ()
		sigma : ()
		sigmas : {}

	xpos : ()					# USED FOR SINGLE FIBRE
	ypos : ()					# USED FOR SINGLE FIBRE

"""


import logging
import numpy as np
import parse

from astropy.io	   import fits
from pyFU.defaults import pyFU_default_keywords, pyFU_default_formats, pyFU_logging_level, pyFU_logging_format

logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)


def header2config (header, keywords=None, formats=None) :
	"""
	Convert a pyFU-compatible FITS header into a pyFU configuration dictionary using
	a dictionary of FITS keyword keys, e.g.

		{'index': 'IF-INDEX', 'xpos-format':'IF-XPFMT'}

	The particular configuration parameters are placed at the highest level in
	the configuration dictionary; the collection of parameters are placed in their
	corresponding sub-dictionaries.
	"""

	if keywords is None :
		keywords = pyFU_default_keywords
	if formats is None :
		formats = pyFU_default_formats

	difu = {}
	dpos = {}
	damp = {}
	dtrace = {'amplitudes':damp, 'positions':dpos}
	dformats = {}
	config = {'ifu':difu,'trace':dtrace,'formats':dformats}
	indices = {}
	labels = {}

	# GET INDEX OF PARTICULAR FIBRE
	idx = None
	if 'index' in keywords and keywords['index'][0] in header :
		idx = header[keywords['index'][0]]
		config['index'] = idx
		indices[idx] = idx
	elif 'trace_index' in keywords and keywords['trace_index'][0] in header :
		idx = header[keywords['trace_index'][0]]
		config['index'] = idx
		indices[idx] = idx
	# GET INDICES OF ALL FIBRES
	elif 'index_format' in keywords and keywords['index_format'][0] in header :
		fmtkey = keywords['index_format'][0]
		fmt = header[fmtkey]
		for i in range(999) :
			key = fmt.format(i)
			if key in header :
				indices[i] = header[key]

	# GET LABEL OF PARTICULAR FIBRE
	flabel = None
	if 'label' in keywords and keywords['label'][0] in header :
		if idx is not None :
			flabel = header[keywords['label'][0]]
			config['label'] = flabel
			labels[idx] = flabel

	# GET LABELS OF ALL FIBRES
	elif 'label_format' in keywords and keywords['label_format'][0] in header :
		fmtkey = keywords['label_format'][0]
		fmt = header[fmtkey]
		for i in range(999) :
			key = fmt.format(i)
			if key in header :
				labels[i] = header[key]

	# RE-CONSTRUCT SLIT
	if len(indices) > 0 :
		l = list(indices.keys())
		l.sort()
		slit = []
		for i in l :
			if i in indices and i in labels :
				slit.append(labels[i])
		difu['slit_labels'] = slit

	# GET POSITION OF PARTICULAR FIBRE
	pos = None
	if 'xpos' in keywords and 'ypos' in keywords and keywords['xpos'][0] in header and keywords['ypos'][0] in header :
		if idx is not None :
			xpos = header[keywords['xpos'][0]]
			ypos = header[keywords['ypos'][0]]
			config['xpos'] = xpos
			config['ypos'] = ypos
			pos = [xpos,ypos]
			difu['focal_positions'] = {labels[idx]:pos}

	# GET POSITIONS OF ALL FIBRES
	elif ('xpos_format' in keywords) and (keywords['xpos_format'][0] in header) \
	 and ('ypos_format' in keywords) and (keywords['ypos_format'][0] in header) :
		xfmtkey = keywords['xpos_format'][0]
		yfmtkey = keywords['ypos_format'][0]
		xfmt = header[xfmtkey]
		yfmt = header[yfmtkey]
		info = config['ifu']
		d = {}
		for i in range(999) :
			xkey = xfmt.format(i)
			ykey = yfmt.format(i)
			if xkey in header and ykey in header :
				x = header[xkey]
				y = header[ykey]
				pos = [x,y]
				if i in labels :
					d[labels[i]] = pos
		difu['focal_positions'] = d
					
	# GET SIGMA OF PARTICULAR FIBRE
	d_in = None
	if 'sigma' in keywords and keywords['sigma'][0] in header :
		key = keywords['sigma'][0]
		sigma = header[key]
		config['sigma'] = sigma
		dtrace['sigmas'] = {labels[idx]:sigma}

	# GET SIGMAS OF ALL FIBRES
	elif 'sigma_format' in keywords and keywords['sigma_format'][0] in header :
		fmtkey = keywords['sigma_format'][0]
		fmt = header[fmtkey]
		d = {}
		for i in range(999) :
			key = fmt.format(i)
			if key in header :
				sigma = header[key]
				if i in labels :
					d[labels[i]] = sigma
		dtrace['sigmas'] = d

	# GET DIAMETER OF PARTICULAR FIBRE
	d_in = None
	if 'inner_diameter' in keywords and keywords['inner_diameter'][0] in header :
		key = keywords['inner_diameter'][0]
		d_in = header[key]
		config['inner_diameter'] = d_in
		difu['inner_diameters'] = {labels[idx]:d_in}

	# GET DIAMETERS OF ALL FIBRES
	elif 'inner_diameter_format' in keywords and keywords['inner_diameter_format'][0] in header :
		fmtkey = keywords['inner_diameter_format'][0]
		fmt = header[fmtkey]
		d = {}
		for i in range(999) :
			key = fmt.format(i)
			if key in header :
				d_in = header[key]
				if i in labels :
					d[labels[i]] = d_in
		difu['inner_diameters'] = d

	d_out = None
	if 'outer_diameter' in keywords and keywords['outer_diameter'][0] in header :
		key = keywords['outer_diameter'][0]
		d_out = header[key]
		config['outer_diameter'] = d_out
		difu['outer_diameters'] = {labels[idx]:d_out}

	elif 'outer_diameter_format' in keywords and keywords['outer_diameter_format'][0] in header :
		fmtkey = keywords['outer_diameter_format'][0]
		fmt = header[fmtkey]
		d = {}
		for i in range(999) :
			key = fmt.format(i)
			if key in header :
				d_out = header[key]
				if i in labels :
					d[labels[i]] = d_out
		difu['outer_diameters'] = d

	# GET TRACE COEFFICIENTS
	if 'trace_format' in keywords and keywords['trace_format'][0] in header :
		fmt = header[keywords['trace_format'][0]]
		dtrace['trace_format'] = fmt

		# ... FOR NAMED FIBRE
		if idx is not None and flabel is not None :
			p = []
			for k in range(9) :
				key = fmt.format(idx,k)
				if key in header :
					p.append(header[key])
			if len(p) == 0 :
				logging.error ('could not extract position coefficients from header for #{0} = {1}'.format(idx,neame))
			else :
				dpos['by_label'] = {flabel:p}
				dpos['by_index'] = {idx:p}
		# ... FOR ALL FIBRES
		elif len(indices) > 0 :
			dpos['by_label'] = {}
			dpos['by_index'] = {}
			for i in indices :
				if i in labels :
					label = labels[i]
					p = []
					for k in range(9) :
						key = fmt.format(i,k)
						if key in header :
							p.append(header[key])
					if len(p) == 0 :
						logging.warning ('could not extract position coefficients from header for #{0} = {1}'.format(i,label))
					else :
						if label is not None :
							dpos['by_label'][label] = p
							dpos['by_index'][i] = p
				else :
					logging.error ('index #{0} is not labelled in header!')
		dtrace['trace_order'] = len(p)-1
	else :
		logging.info('no "trace_format" equivalent keyword in header')

	# GET AMPLITUDE MODEL
	if 'ampl_model' in keywords and keywords['ampl_model'][0] in header :
		key = keywords['ampl_model'][0]
		dtrace['ampl_model'] = header[key]
		# print (dtrace['ampl_model'])
	else :
		logging.warning ('no "ampl_model" keyword in header')

	# GET AMPLITUDE COEFFICIENTS
	if 'ampl_format' in keywords and keywords['ampl_format'][0] in header :
		fmt = header[keywords['ampl_format'][0]]
		dtrace['ampl_format'] = fmt

		# ... FOR NAMED FIBRE
		if idx is not None and flabel is not None :
			p = []
			for k in range(16) :
				key = fmt.format(idx,k)
				if key in header :
					p.append(header[key])
			if len(p) == 0 :
				logging.error ('could not extract ampl coefficients from header for #{0} = {1}'.format(idx,neame))
			else :
				damp['by_label'] = {flabel:p}
				damp['by_index'] = {idx:p}
		# ... FOR ALL FIBRES
		elif len(indices) > 0 :
			damp['by_label'] = {}
			damp['by_index'] = {}
			for i in indices :
				if i in labels :
					label = labels[i]
					p = []
					for k in range(9) :
						key = fmt.format(i,k)
						if key in header :
							p.append(header[key])
					if len(p) == 0 :
						logging.warning ('could not ampl coefficients from header for #{0} = {1}'.format(i,label))
					else :
						if label is not None :
							damp['by_label'][label] = p
							damp['by_index'][i] = p
				else :
					logging.error ('index #{0} is not labelled in header!')
		dtrace['ampl_order'] = len(p)-1
	else :
		logging.info ('no "ampl_format" equivalent keyword in header')

	# RETURN CONFIGURATION DICTIONARY
	return config


class HeaderFilter (object) :
	"""
	Performs various filtering of FITS headers to extract or filter out pyFU info.
	"""
	def __init__ (self, header=None, keywords=None, formats=None) :
		if keywords is None :
			keywords = pyFU_default_keywords
		if formats is None :
			formats = pyFU_default_formats

		self.keywords = keywords
		self.formats  = formats
		self.keep     = []		# NORMALLY KEPT
		self.ignore   = []		# NORMALLY IGNORED
		self.prefixes = []		# LIST OF FORMAT PREFIXES AND FORMATS
		self.nuke     = []		# LIST OF ALL POSSIBLE pyFU KEYWORDS

		# GET AND KEEP ALL FORMAT CARDS
		if keywords is not None :
			for key,fmt in pyFU_default_formats.items() :	# FOR ALL POSSIBLE FORMAT KEYS
				if key in keywords :									# keywords HAS keyw,comment
					keyw = keywords[key][0]								# FITS KEYWORD FOR THE FORMAT
					if (header is not None) and (keyw in header) :		# PREFERABLY USE HEADER FORMATS
						fmt = header[keyw]
					elif (formats is not None) and (key in formats) :	# ... OR USER ONES
						fmt = formats[key]
						self.keep.append(keyw)
						self.nuke.append(keyw)
					if len(fmt) == 2 : fmt = fmt[0]	# GET RID OF DESCRIPTIONS
					self.prefixes.append ((fmt[:fmt.index('{')],fmt))

		# STANDARD GENERIC THINGS TO IGNORE START WITH...
		self.ignore = ['SIMPLE','EXTEND','BITPIX','COMMENT','XTENSION',
					'NAXIS','CTYPE','CUNIT','CRPIX','END',
					'PCOUNT','TFIELDS','TFORM','TTYPE','TUNIT']

		# IGNORE OTHER NON-INDEXED pyFU KEYWORDS ENTIRELY?
		for key in self.keywords :
			keyw,comment = self.keywords[key]
			if keyw is not None :
				self.nuke.append(keyw)

	def copy_header (self, hdr_in,hdr_out, idx=None, bare=False, comments=True) -> bool :
		"""
		Copies non-fundamental header entries from one FITS header to another.
		If idx and keywords are not None, then only the pyFU metadata for that spectrum are transferred.

		The spectrum index is assumed to be placed before any polynomial order in the coded info.
		The "keywords" dictionary is assumed to have the structure {key: (keyword,comment), ...}.
		The "formats" dictionary is assumed to have the structure {key: format,, ...}.

		If the "bare" option is chosen, all pyFU keywords are excluded.
		"""
		if (hdr_in is None) or (hdr_out is None) :
			return False

		# CHECK ALL KEYWORDS IN INPUT HEADER AND TRANSFER THOSE THAT ARE WANTED
		for keyw in hdr_in :
			ok = True

			# REMOVE GENERALLY IGNORED KEYWORDS
			for ignkeyw in self.ignore :
				if keyw.startswith(ignkeyw) :
					ok = False
					# print (keyw,ignkeyw,ok)

			# REMOVE INDEXED KEYWORDS WITH WRONG INDEX
			if idx is not None :
				for prfx,fmt in self.prefixes :
					if keyw.startswith(prfx) :	# COULD BE...
						things = sscanf(fmt,keyw)
						if len(things) >= 1 and things[0] == idx :
							ok = True
							# print (keyw,fmt,things,' == ',idx,ok)
						else :
							ok = False
							# print (keyw,fmt,things,' != ',idx,ok)


			# REMOVE ALL pyFU KEYWORDS
			if bare :
				if keyw in self.nuke :	# NON-INDEXED KEYWORDS
					ok = False
				else :					# TEST FOR INDEXED KEYWORD
					for prfx,fmt in self.prefixes :
						if keyw.startswith(prfx) :	# COULD BE...
							things = sscanf(fmt,keyw)
							if len(things) > 0 :	# YUP!
								ok = False
							# print (keyw,fmt,things,ok)

			# OR KEEP KEPT KEYWORDS
			else :
				if keyw in self.keep :
					ok = True
					# print ('keep',keyw,ok)

			# TRANSFER
			if ok :
				val = hdr_in[keyw]
				if isinstance (hdr_in,fits.Header) :
					if comments :
						logging.debug (f'{idx}: transferring {keyw}={val} / {hdr_in.comments[keyw]}')
						hdr_out[keyw] = (val,hdr_in.comments[keyw])
					else :
						hdr_out[keyw] = val
				else :
					try :
						hdr_out[keyw] = val
					except ValueError as e :
						logging.warning (str(e))

		return True

def stripped_format (fmt) :
	"""
	Takes a perfectly reasonable python format string and removes the size info so that
	parse.parse can read values using it (like C's "sscanf").
	Any reasonable language having str.format would have the reverse operation..... :-(

	Example:  "COE{0:03d}_{1:1d}"...
	"""
	parts = fmt.split('{')	# e.g. ["COE","0:03d}_","1:1d}"]
	if len(parts) == 1 :
		raise ValueError ('string is not a format: {0}'.format(fmt))
	f = parts[0]
	for i in range(1,len(parts)) :
		if '}' not in parts[i] :	# WHOOPS!
			raise ValueError ('no matching {} in format!')
		things = parts[i].split('}')		# e.g. ["0:03d","_"]
		stuff = things[0].split(':')		# e.g. ["0","03d"]
		f += '{:'+stuff[1]+'}'
		if len(things) == 2 :
			f += things[1]
	return f

def sscanf (fmt,s) :
	"""
	Rough equivalent of C's sscanf using parse.parse.
	Returns a list of values extracted using a format string.
	"""
	try :
		things = parse.parse (stripped_format(fmt),s).fixed
		return things
	except :
		return []

def main () :
	logging.critical ('meta.py does not have a working main () method!')

