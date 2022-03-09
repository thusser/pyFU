#!/usr/bin/env python3

# display.py

"""
A version with failed cursor input is still contained here....  :-(
matplotlib is a mess w.r.t. this sort of stuff!                >:-O
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from utils         import get_image_limits, show_hdu
from astropy.io    import fits

from pyFU.defaults import pyFU_default_formats, pyFU_default_keywords
from pyFU.utils    import get_infiles_and_outfiles

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)

def plot_tables (tables, idx=None, label=None, key=None, value=None,
				xkey='wavelength', ykey='flux', errkey=None,
				xlabel=None, ylabel=None, title=None, flxmin=0, flxmax=None, 
				show_metadata=False, mode='individual', outfile=None,
				*args, **kwargs) :
	"""
	Plots spectra from a set of tables.

	If idx != None, then only the idx-th table is plotted (note 1 <= idx <= n).

	If label != None, then only the table with that fibre label is plotted.

	If key and value are not None, then only the tables with the metadata key=value are plotted.
	The keyword can contain a "*" to indicate partial matching, e.g. "ID-LA*" for "ID-LA123".

	If mode != 'individual', then the spectra are plotted on the same plot.
	"""
	# plt.clf()
	n = len(tables)
	label_format = pyFU_default_formats['label_format']
	index_key    = pyFU_default_keywords['index'][0]

	# POTENTIALLY PLOT ALL SPECTRA?
	if idx is None :
		indices = range(1,n+1)

	# OR JUST SPECTRUM WITH INDEX idx
	else :
		if idx >= 1 and idx <= n :
			indices = [idx]
			mode = 'individual'
		else :
			logging.error (f'spectrum index={idx} > number of spectra ({n})')
			return

	# START GROUP PLOT 
	if mode != 'individual' :
		fig = plt.figure ()
		# plt.style.use ('ggplot')

	# FOR ALL TABLES
	nplotted = 0
	for i in indices :
		table = tables[i-1]
		hdr = table.meta

		# GET FIBRE LABEL FROM INTERNAL INDEX
		labl = None
		if index_key in hdr :
			indx = hdr[index_key]
			ky = label_format.format(indx)
			labl = hdr[ky]

		# SHOW METADATA?
		if show_metadata :
			logging.info (f'Metadata for table #{i}:')
			for ky in hdr :
				logging.info (f'\t{ky} = {hdr[ky]}')

		# PLOT OR NOT?
		ok = False

		# PLOT SPECTRUM WITH A GIVEN int INDEX
		if idx is not None and idx == i :					# RIGHT INDEX
			ok = True

		# FIND SPECTRUM WITH THE RIGHT key=value COMBINATION?
		elif (not key is None) and (not value is None) :	# RIGHT KEYWORD AND VALUE
			keys = [key]
			if key.startswith('*') :						# FIND ALL KEYS THAT START WITH key
				for ky in hdr :
					if ky.endswith(key[1:]) :
						keys.append(ky)
			elif key.endswith('*') :						# FIND ALL KEYS THAT END WITH key
				for ky in hdr :
					if ky.startswith(key[:-1]) :
						keys.append(ky)
			elif '*' in key :
				keystart = key[:key.index('*')]
				keyend   = key[key.index('*')+1:]
				for ky in hdr :
					if ky.startswith(keystart) and ky.endswith(keyend) :
						keys.append(ky)
			for ky in keys :								# FIND ALL HEADER KEYS THAT MATCH
				if (ky in hdr) and (hdr[ky] == value) :
					logging.debug (f'{ky} == {value}')
					ok = True

		# SPECTRUM WITH THE RIGHT LABEL?
		elif label is not None and label == labl :
			ok = True

		# PLOT ANY
		else :
			ok = True

		if ok :
			if mode == 'individual' :
				fig = plt.figure ()
				# plt.style.use ('ggplot')

			# CHECK TO SEE IF THE TABLE HAS THE RIGHT xkey : IF NOT, USE "pixel"
			if not xkey in table.colnames :
				xkey = 'pixel'

			try :
				if errkey is None :
					plt.plot (table[xkey],table[ykey],'-', *args, **kwargs)
				else :
					plt.errorbar (table[xkey],yerr=table[errkey], *args, **kwargs)
				nplotted += 1
			except KeyError as e :
				logging.error ('cannot access table column(s):'+str(e))
				logging.warning ('available columns:'+str(table.colnames))
				return

			if mode == 'individual' :
				if flxmin is not None :
					plt.ylim (bottom=flxmin)
				if flxmax is not None :
					plt.ylim (top=flxmax)
				plt.xlabel (xkey)
				plt.ylabel (ykey)
				if xlabel is not None : plt.xlabel (xlabel)
				if ylabel is not None : plt.ylabel (ylabel)
				if title is None : 
					plt.title (f'{title}, fibre "{labl}" = #{i}/{n}')
				else :
					plt.title (f'fibre "{labl}" = #{i}/{n}')

				logging.info (f'mean of spectrum #{i} is {np.nanmean(table[ykey]):.2f}')

				# plt.tight_layout ()
				if outfile is None :
					if show_with_menu (fig,['abort']) == 'abort' :
						return
				else :
					plt.savefig (outfile)

		# END OF ALL TABLES

	# FINISH OFF GROUP PLOT
	if nplotted > 0 and mode is not None and mode != 'individual' :
		if flxmin is not None :
			plt.ylim (bottom=flxmin)
		if flxmax is not None :
			plt.ylim (top=flxmax)
		plt.xlabel (xkey)
		plt.ylabel (ykey)
		if xlabel is not None : plt.xlabel (xlabel)
		if ylabel is not None : plt.ylabel (ylabel)
		if title  is not None : plt.title  (title)
		# plt.tight_layout ()
		if outfile is None :
			plt.show ()
		else :
			plt.savefig (outfile)

def cursor_input (data, show_method=show_hdu, *args, **kwargs) :
	"""
	Experimental method to display the data from a FITS HDU and then permit the user to measure
	points using the cursor.  Default display is of an image.
	"""
	show_method (data, *args, **kwargs)
	plt.setp(plt.gca(), autoscale_on=False)

	pts = []
	pt = plt.ginput(1, timeout=-1, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
	while len(pt) == 1 :
		pts.append(pt)
		pt = plt.ginput(1, timeout=-1, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=2)
	return pts

def show_hex (hdus, xkey='IFU-XPOS', ykey='IFU-YPOS', wavelength=None) :
	"""
	Displays a set of IFU spectra contained in a FITS HDU table list as an image.
	The positions of the fibres are read from the headers.
	If no wavelength is given, the flux integrated over all wavelengths is used.
	"""
	pass


"""
Taken directly from the Matplotlib Widgets Menu example
https://matplotlib.org/gallery/widgets/menu.html#sphx-glr-gallery-widgets-menu-py
"""

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.mathtext as mathtext
import matplotlib.artist as artist
import matplotlib.image as image

class ItemProperties (object):
	def __init__(self, fontsize=14, labelcolor='black', bgcolor='gray',
				 alpha=1.0):
		self.fontsize = fontsize
		self.labelcolor = labelcolor
		self.bgcolor = bgcolor
		self.alpha = alpha
		self.labelcolor_rgb = colors.to_rgba(labelcolor)[:3]
		self.bgcolor_rgb = colors.to_rgba(bgcolor)[:3]

class SimpleMenuItem (artist.Artist):
	parser = mathtext.MathTextParser("Bitmap")
	padx = 5
	pady = 5
	def __init__(self, fig, labelstr, props=None, hoverprops=None, on_select=None):
		artist.Artist.__init__(self)
		self.set_figure(fig)
		self.labelstr = labelstr
		if props is None:
			props = ItemProperties()
		if hoverprops is None:
			hoverprops = ItemProperties()
		self.props = props
		self.hoverprops = hoverprops
		self.on_select = on_select
		x, self.depth = self.parser.to_mask(
			labelstr, fontsize=props.fontsize, dpi=fig.dpi)
		if props.fontsize != hoverprops.fontsize:
			raise NotImplementedError(
				'support for different font sizes not implemented')
		self.labelwidth = x.shape[1]
		self.labelheight = x.shape[0]
		self.labelArray = np.zeros((x.shape[0], x.shape[1], 4))
		self.labelArray[:, :, -1] = x/255.
		self.label = image.FigureImage(fig, origin='upper')
		self.label.set_array(self.labelArray)
		self.rect = patches.Rectangle((0, 0), 1, 1)
		self.set_hover_props(False)
		fig.canvas.mpl_connect('button_release_event', self.check_select)

	def check_select(self, event):
		over, junk = self.rect.contains(event)
		if not over:
			return
		if self.on_select is not None:
			self.on_select(self)

	def set_extent(self, x, y, w, h):
		self.rect.set_x(x)
		self.rect.set_y(y)
		self.rect.set_width(w)
		self.rect.set_height(h)
		self.label.ox = x + self.padx
		self.label.oy = y - self.depth + self.pady/2.
		self.hover = False

	def draw(self, renderer):
		self.rect.draw(renderer)
		self.label.draw(renderer)

	def set_hover_props(self, b):
		if b:
			props = self.hoverprops
		else:
			props = self.props
		r, g, b = props.labelcolor_rgb
		self.labelArray[:, :, 0] = r
		self.labelArray[:, :, 1] = g
		self.labelArray[:, :, 2] = b
		self.label.set_array(self.labelArray)
		self.rect.set(facecolor=props.bgcolor, alpha=props.alpha)

	def set_hover(self, event):
		'''check the hover status of event and return true if status is changed'''
		b, junk = self.rect.contains(event)
		changed = (b != self.hover)
		if changed:
			self.set_hover_props(b)
		self.hover = b
		return changed

class SimpleMenu(object):
	def __init__(self, fig, menuitems):
		self.figure = fig
		fig.suppressComposite = True
		self.menuitems = menuitems
		self.numitems = len(menuitems)
		maxw = max(item.labelwidth for item in menuitems)
		maxh = max(item.labelheight for item in menuitems)
		totalh = self.numitems*maxh + (self.numitems + 1)*2*SimpleMenuItem.pady
		x0 = 10
		y0 = 70
		width = maxw + 2*SimpleMenuItem.padx
		height = maxh + SimpleMenuItem.pady
		for item in menuitems:
			left = x0
			bottom = y0 - maxh - SimpleMenuItem.pady
			item.set_extent(left, bottom, width, height)
			fig.artists.append(item)
			y0 -= maxh + SimpleMenuItem.pady
		fig.canvas.mpl_connect('motion_notify_event', self.on_move)

	def on_move(self, event):
		draw = False
		for item in self.menuitems:
			draw = item.set_hover(event)
			if draw:
				self.figure.canvas.draw()
				break

show_result = None

def show_menu_reaction (item):
	global show_result
	print('You selected "%s"' % item.labelstr)
	show_result = item.labelstr

def show_with_menu (fig, items, on_select=show_menu_reaction, left=0.2, bottom=0.2, top=0.9, right=0.9) :
	"""
	To test:
		fig = plt.figure ()
		plt.plot (range(10),range(10),'o')
		r = show_with_menu (fig, ['do this','do that'])
		print (r)
	"""
	global show_result
	fig.subplots_adjust(left=left,bottom=bottom,top=top,right=right)
	props = ItemProperties(labelcolor='black', bgcolor='gray', fontsize=12, alpha=0.2)
	hoverprops = ItemProperties(labelcolor='white', bgcolor='blue', fontsize=12, alpha=0.2)
	menuitems = []
	for label in items :
		item = SimpleMenuItem(fig, label, props=props, hoverprops=hoverprops, on_select=on_select)
		menuitems.append(item)
	if len(menuitems) > 0 :
		menu = SimpleMenu(fig, menuitems)
	plt.show ()
	plt.close ()
	return show_result

def main () :
	import argparse
	import sys
	from pyFU.utils import read_tables, initialize_logging
	from astropy.io import fits
	from pyFU.trace import SpectrumTracer

	# ---- PARSE COMMAND LINE
	parser = argparse.ArgumentParser ()
	parser.add_argument ('--fibre','-F',default=None,type=str,
		help='plot spectrum with this fibre label')
	parser.add_argument ('--flip','-f',default=False,action='store_true',
		help='show images with first row on top (default False, i.e. first row on bottom)')
	parser.add_argument ('--hdu','-H',default=0,type=int,
		help='image HDU to display (default 0)')
	parser.add_argument ('--images','-i',default=None,
		help='path(s) of input FITS image(s)')
	parser.add_argument ('--index','-I',default=None,type=int,
		help='index of spectrum (1,..,number)')
	parser.add_argument ('--labeled','-l',default=False,action='store_true',
		help='label the fibres (default False)')
	parser.add_argument ('--key','-K',default=None,
		help='FITS keyword for special selection (see --value) (default None; can use * as wildcard)')
	parser.add_argument ('--metadata','-m',default=False,action='store_true',
		help='print out FITS header (default False)')
	parser.add_argument ('--nobar','-n',default=False,action='store_true',
		help='do not add colourbar (default False)')
	parser.add_argument ('--outfiles','-o',default=None,
		help='path(s) of output JPG,PNG,... image(s)')
	parser.add_argument ('--xcol','-X',default='wavelength',
		help='name of x-column in table')
	parser.add_argument ('--ycol','-Y',default='flux',
		help='name of y-column in table')
	parser.add_argument ('--tables','-t',default=None,
		help='path(s) of input binary FITS table(s)')
	parser.add_argument ('--value','-V',default=None,
		help='FITS keyword value for special selection (see --key) (default None)')
	parser.add_argument ('--width','-w',type=int,default=None,
		help='display int width of given extraction band (default None)')
	parser.add_argument ('--yaml','-y',default=None,
		help='path of yaml configuration file (default None)')
	parser.add_argument ('--zmax','-Z',default=None,
		help='upper displayed value')
	parser.add_argument ('--zmin','-z',default=None,
		help='lower displayed value')
	args = parser.parse_args ()

	# ---- INTIALIZE LOGGING
	initialize_logging (config_file=args.yaml)

	# ---- GET LIST OF TABLES/IMAGES
	if args.tables is not None :
		infiles,outfiles = get_infiles_and_outfiles (args.tables,args.outfiles)
		for infile,outfile in zip(infiles,outfiles) :
			spectra,hdr = read_tables (infile)
			if args.metadata :
				logging.info (f'\n\tMetadata for tables:')
				for key in hdr :
					logging.info (f'\t{key} = {hdr[key]}')
			plot_tables (spectra,idx=args.index,key=args.key,value=args.value,label=args.fibre, \
						mode='individual', xkey=args.xcol,ykey=args.ycol,title=infile, \
						show_metadata=args.metadata, outfile=outfile)
			if args.index is None :
				plt.title (infile)
				if outfile is None :
					plt.show()
				else :
					print ('Saving image file',outfile,'...')
					plt.savefig (outfile)

	elif args.images is not None :
		infiles,outfiles = get_infiles_and_outfiles (args.images,args.outfiles)
		for infile,outfile in zip(infiles,outfiles) :
			hdus = fits.open (infile)
			hdu = hdus[args.hdu]
			hdr = hdu.header
			comments = hdr.comments
			logging.debug (f'\tnumber of HDU in FITS file:{len(hdus)}')
			if args.metadata :
				print (f'\theader of HDU #{args.hdu+1}:')
				for key in hdr :
					print (f'\t\t{key} = {hdr[key]} / {comments[key]}')

			if 'CUNIT1' in hdr :
				plt.xlabel (hdr['CUNIT1'])
			if 'CUNIT2' in hdr :
				plt.ylabel (hdr['CUNIT2'])

			show_hdu (hdu,aspect=None,colourbar=(not args.nobar), \
							vmin=args.zmin,vmax=args.zmax)
			if args.labeled :
				tracer = SpectrumTracer (hdu)
				tracer.plot_traces (title=infile, width=args.width, outfile=outfile)
			if outfile is None :
				plt.show()
			else :
				print ('Saving image file',outfile,'...')
				plt.savefig (outfile)

	else :
		print ('No input data!')
		sys.exit(1)

	sys.exit(0)
	"""
	print ('version with cursor input:')
	pts = cursor_input (hdus[0])
	print (pts)
	"""

if __name__ == '__main__' :
	main ()
