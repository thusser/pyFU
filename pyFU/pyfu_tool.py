#!/opt/local/bin/python

# pyfu_tool.py

"""
Command-line tool to test the pyFU pipeline functionality or to perform instrument tests.
Assumes the pyFU package has been so installed that the "ifu*" commands are in the current path.

Version 0.4; F.V. Hessman (2022-FEB-23)

- Example of reducing many raw images :

		000> bias  ./bias/*.fits ./calib/masterbias.fits  # CREATE MASTER BIAS
		001> dark  ./dark/*.fits ./calib/unitdark.fits    # CREATE UNIT DARK
		002> flat  ./flat/*.fits ./calib/masterflat.fits  # CREATE MASTER FLAT
		003> calib ./raw/*.fits  ./calib/red_*.fits       # REDUCE TO CALIBRATED IMAGES

- Example of quick IFU imaging

		001> set show false                 # DON'T BOTHER TO TAKE A LOOK AT THE INTERMEDIATE RESULTS
		002> set trace_cfg ./mytracing.yaml # GOOD CONFIGURATION NEEDED FOR SUCCESSFUL TRACING
		003> trace   ./myimage.fits
		004> extract ./myimage.fits ./extracted.fits
		005> pimage  ./extracted.fits

- Example of setting trace parameters (e.g. from bright star)

		001> set show true        # VIEW ALL OF THE GORY DETAILS OF THE TRACING
		002> set trace_cfg none   # RESET IN CASE OF PREVIOUS ENTRY
		003> calib ./raw/raw_bright.fits ./calib/reduced_bright.fits
		004> trace ./calib/reduced_bright.fits ./calib/bright_trace.yaml
		005> set trace_cfg ./calib/bright_trace.yaml

- Example of full reduction with pre-determined trace parameters

		001> set show false
		002> set trace_cfg ./calib/bright_trace.yaml           # WHERE THE TRACE INFO COMES FROM
		003> calib   ./raw_00123.fits.gz ./pipe/reduced.fits   # REDUCE RAW IMAGE
		004> extract ./pipe/reduced.fits ./pipe/extracted.fits # EXTRACT USING PREVIOUS TRACE INFO
		005> wavcal  ./pipe/extracted.fits ./pipe/cal_00123.fits.gz
"""

generic = \
"""
# general pyFU configuration file info
logging:
  level: info # | debug
  format:
  file:                     # OPTIONAL HARD COPY
ifu:
  name: myIFU
  number_fibres: 2          # NUMBER OF FIBRES >= NUMBER OF SPECTRA IN IFU SPECTRAL IMAGE
  slit_labels: [eg1,eg2]    # LIST OF FIBRE LABELS/NAMES
  slit_start: 1             # WHICH LABEL IS THE STARTING LABEL ACTUALLY USED? [1..number_fibres]
  inner_diameter: 100       # GENERIC INNER DIAMETER OF FIBRES (MICRONS)
  outer_diameter: 110       # GENERIC OUTER DIAMETER OF FIBRES (MICRONS)
  inner_diameters:          # SPECIFIC INNER DIAMETERS OF NAMED FIBRES
    eg1: 101
    eg2: 102 
  outer_diameters:          # SPECIFIC OUTER DIAMETERS OF NAMED FIBRES
    eg1: 111
    eg2: 112
  focal_positions:          # (MICRONS)
    eg1: [-100.,0.]
    eg2: [+100.,0.]
"""

import logging
import os
import sys
import yaml

from astropy.io    import fits
from pyFU.defaults import pyFU_logging_format
from pyFU.utils    import initialize_logging

cfg = {}

def load_tool_cfg () :
	global cfg
	if os.path.exists ('./.pyfu_tool') :
		with open('./.pyfu_tool','r') as stream :
			cfg = yaml.safe_load (stream)
	if 'calib_img' not in cfg :
		cfg['calib_img'] = None
	if 'extr_img' not in cfg :
		cfg['extr_img'] = None
	if 'logging' not in cfg :
		cfg['logging'] = {'level':logging.INFO,'format':pyFU_logging_format,'file':'./.pyfu_tool.log'}
	if 'masterbias' not in cfg :
		cfg['masterbias'] = None
	if 'masterflat' not in cfg :
		cfg['masterflat'] = None
	if 'pyfu_cfg' not in cfg :
		cfg['pyfu_cfg'] = None
	if 'show' not in cfg :
		cfg['show'] = False 
	if 'trace_cfg' not in cfg :
		cfg['trace_cfg'] = None
	if 'trace_img' not in cfg :
		cfg['trace_img'] = None
	if 'unitdark' not in cfg :
		cfg['unitdark'] = None
	if 'wavcal_img' not in cfg :
		cfg['wavcal_img'] = None
	if 'working' not in cfg :
		cfg['working'] = '.'
	initialize_logging (config=cfg)

def save_tool_cfg () :
	global cfg
	with open ('./.pyfu_tool','w') as stream :
		yaml.dump (cfg,stream)

def status () :
	global cfg
	print ('\nConfiguration:')
	print (f'\tlogging    {cfg["logging"]}')
	print (f'\tmasterbias {cfg["masterbias"]}\t\t\t\t(path of masterbias image)')
	print (f'\tmasterflat {cfg["masterflat"]}\t\t\t\t(path of masterflat image)')
	print (f'\tpyfu_cfg   {cfg["pyfu_cfg"]}\t\t\t\t(pyFU configuration file)')
	print (f'\tshow       {cfg["show"]}\t\t\t\t(live plotting option; 0|1|false|true)')
	print (f'\ttrace_cfg  {cfg["trace_cfg"]}\t\t\t\t(path of trace configuation file)')
	print (f'\tunitdark   {cfg["unitdark"]}\t\t\t\t(path of unit dark image)')
	print (f'\tworking    {cfg["working"]}\t\t\t\t(working directory)')
	print ('\nLast images:')
	print (f'\tlast calibrated image            : {cfg["calib_img"]}') 
	print (f'\tlast traced image                : {cfg["trace_img"]}') 
	print (f'\tlast extracted image             : {cfg["extr_img"]}') 
	print (f'\tlast wavelength calibrated image : {cfg["trace_img"]}') 

def help () :
	print ('\tcommand    parameter#1                                   parameter#2\n')
	print ('\t-------    -----------                                   -----------')
	print ('\t$          {shell command}')
	print ('\tba{sic}    {name of configuration file to be created}')
	print ('\tbi{as}     {name of masterbias or bias images directory} {optional name of masterbias}')
	print ('\tca{lib}    {name of raw image file}                      {name of calibrated image file}')
	print ('\tcr{eate}   {name of fake image to create using configuration file}')
	print ('\tco{nfig}   {name of pyfu YAML configuration file}')
	print ('\tda{rk}     {name of unitdark or dark images directory}   {optional path of unitdark}')
	print ('\tex{tract}  {name of image file}                          {optional path of extracted table}')
	print ('\tfh{eader}  {name of FITS file}')
	print ('\tfl{at}     {name of masterflator flat images directory}  {optional path of masterbias}')
	print ('\the{lp}')
	print ('\tid{isplay} {name of image file to display}')
	print ('\tlo{gging}  {debug,info}')
	print ('\tpi{mage}   {pathname of table to be imaged}              {optional path of PNG image integrated in pixels}')
	print ('\tpl{ot}     {name of table file}                          {index of spectrum}')
	print ('\ttd{isplay} {name of table file to display}')
	print ('\tq{uit}')
	print ('\tse{t}      {name of configuration parameter}             {new value}')
	print ('\tst{atus}')
	print ('\ttr{ace}    {pathname of image to be traced}              {optional path of trace configuration file}')
	print ('\twi{mage}   {pathname of table to be imaged}              {optional path of PNG image integrated in wavelength}')
	print ('\two{rking}  {pathname of working directory}')
	print ('')
	print ('\t{output} = {input1} +|-|*|/|^ {input2}')
	print ('\t{output} = {abs|mean|median|sum|sqrt|xmean|ymean} {input}')
	print ('\t<RETURN>             (to show old commands and their numbers)')
	print ('\t{number}             (to execute old commands by number)')

def is_fits_file (s) :
	"""
	Tests filenames for .fits or .fit suffix.
	If the string contains a '*' then it is assumed to be a pattern.
	"""
	forms = ['.fits','.fit','.fits.gz','.fit.gz']
	if isinstance(s,str) :
		if '*' in s : return False
		ss = s.lower()
		for f in forms :
			if ss.endswith(f) : return True
	else :
		for x in s :
			ss = x.lower()
			if '*' in ss : return False
			for f in forms :
				if ss.endswith(f) : return True
	return False

def str_is_int (s) :
	try :
		i = int(s)
		return True
	except :
		return False

def fullpath (pathname) :
	""" Adds prefix of working directory if no other hierarchy is given """
	global cfg
	if pathname.startswith ('"') and pathname.endswith ('"') :
		return pathname[1:-2]
	elif pathname.startswith ('/') or pathname.startswith('~') or pathname.startswith('.') or pathname.startswith('..') :
		return pathname
	else :
		return cfg['working']+'/'+pathname

def peruse_cmds (comms) :
	nn = len(comms)
	if nn == 0 : return
	mm = 0
	for ii in range(nn-1,-1,-1) :
		print ('\t',ii,' : ',comms[ii])
		mm += 1
		if mm == 20 :
			res = input('(<RETURN> for more)')
			if res != '' :
				return
			mm = 0
	return

def do_cmd (cmdstring) :
	print ('[',cmdstring,']')
	os.system (cmdstring)

def tool () :
	global cfg
	ok = True
	transID = 1
	cmds = []
	commands   = ['$','basic','bias','calib','create','dark','display','fheader','flat',
			'help','logging','pimage','plot','quit','set','status',
			'trace','wimage']
	load_tool_cfg ()

	print ('\npyFU tool, Version 2022-FEB-25\n===========================\n')
	help ()

	normal_cmd = True
	try :
		while (ok) :
			# GET NEXT COMMAND
			if normal_cmd :
				cmd0 = input (f'{transID:03d}> ')
			else :
				normal_cmd = True

			# PARSE
			cmd = cmd0
			extn = ''
			if cmd != '' :
				parts = cmd0.split()
				nparts = len(parts)
				print (parts)
				if '#' in parts :		# LOOK FOR COMMENTS
					nparts = parts.index('#')
					parts = parts[:nparts]
				if '&' in parts :		# LOOK FOR VERBATIM EXTENSIONS
					nparts = parts.index('&')
					parts = parts[:nparts]
					extn = ' '+cmd0[cmd0.index('&')+2:]
					print (f'extn=[{extn}]')
				cmd = parts[0]

			# PROCESS COMMAND
			if cmd.startswith('q') :	# QUIT
				ok = False
			elif cmd.startswith('he') :	# HELP
				help()
			elif cmd == '$' :
				do_cmd (cmd0.strip()[1:]+extn)
			elif cmd == '' :
				peruse_cmds (cmds)
			elif str_is_int (cmd) and int(cmd) < len(cmds) :
				cmd0 = cmds[int(cmd)]
				print (f'{transID:03d}> {cmd0}')
				normal_cmd = False

			elif cmd.startswith('ba') :
				logging.info ('Creating the generic configuration YAML file "generic.yaml" ...')
				with open ('./generic.yaml','w') as stream :
					stream.write (generic)
				progs = ['ifucal','ifutra','ifuext','ifulam','ifuwav','ifureb','ifuima','ifusol','ifufak']
				for prog in progs :
					ifu_cmd = prog+' -G ./generic.yaml'
					do_cmd (ifu_cmd+extn)
					
			elif cmd.startswith('bi') and nparts > 1 :	# BIAS
				"""
				Command:  bias {name or directory} {optional path of masterbias}
				Function: creates a masterbias
				"""
				infile = parts[1].lower()
				if infile == 'none' :				# a) USED TO AVOID BIAS CORRECTION, or
					cfg['masterbias'] = None
				elif is_fits_file (infile) :		# b) FITS-FILE -> SET masterbias TO THIS FILE, or
					cfg['masterbias'] = fullpath(parts[1])
				else :								# c) DIRECTORY -> CREATE MASTERBIAS
					ifu_cmd = 'ifucal'
					pyfu_cfg = cfg['pyfu_cfg']
					if pyfu_cfg is not None :
						ifu_cmd += f' --yaml {pyfu_cfg}'
					ifu_cmd += f' --bias_files "{fullpath(parts[1])}"'
					if nparts == 2 :
						cfg['masterbias'] = fullpath('masterbias.fits')
					else :
						cfg['masterbias'] = fullpath(parts[2])
					ifu_cmd += f' --masterbias {cfg["masterbias"]}'
					if cfg['show'] :
						ifu_cmd += ' --plot'
					logging.info (ifu_cmd)
					do_cmd (ifu_cmd+extn)
				print ('\tnew masterbias file is',cfg['masterbias'])

			elif cmd.startswith('ca') and nparts >= 2 :	# CALIB
				ifu_cmd = 'ifucal'
				if cfg['pyfu_cfg'] is not None :
					ifu_cmd += f' --yaml {cfg["pyfu_cfg"]}'
				ifu_cmd += f' --infiles "{fullpath(parts[1])}"'
				if cfg['masterbias'] is not None :
					ifu_cmd += f' --subtract_bias --masterbias {cfg["masterbias"]}'
				if cfg['unitdark'] is not None :
					ifu_cmd += f' --subtract_dark --unitdark {cfg["unitdark"]}'
				if nparts == 3 :
					outfile = fullpath(parts[2])
				else :
					outfile = fullpath('reduced.fits')
				ifu_cmd += f' --outfiles "{outfile}"'
				if cfg['show'] :
					ifu_cmd += ' --plot'
				do_cmd (ifu_cmd+extn)
				cfg['calib_img'] = outfile
				print ('\tcalibrated file is',outfile)

			elif cmd.startswith('cr') :	# CREATE
				ifu_cmd = 'ifufak'
				if cfg['pyfu_cfg'] is not None :
					ifu_cmd += f' --yaml {cfg["pyfu_cfg"]}'
				if nparts == 2 :
					ifu_cmd += f' --outfile {parts[1]}'
				do_cmd (ifu_cmd+extn)

			elif cmd.startswith('da') and nparts > 1 :		# DARK
				infile = parts[1].lower()
				if infile == 'none' :
					cfg['unitdark'] = None
				elif is_fits_file (infile) :
					cfg['unitdark'] = fullpath(parts[1])
				else :
					ifu_cmd = 'ifucal'
					if cfg['pyfu_cfg'] is not None :
						ifu_cmd += f' --yaml {cfg["pyfu_cfg"]}'
					if cfg['masterbias'] is not None :
						ifu_cmd += f' --subtract_bias --masterbias {cfg["masterbias"]}'
					ifu_cmd += f' --dark_files "{fullpath(parts[1])}"'
					if nparts == 2 :
						cfg['unitdark'] = fullpath('unitdark.fits')
					else :
						cfg['unitdark'] = fullpath(parts[2])
					ifu_cmd += f' --unitdark {cfg["unitdark"]}'
					if cfg['show'] :
						ifu_cmd += ' --plot'
					do_cmd (ifu_cmd+extn)

			elif (cmd.startswith('id') or cmd.startswith('di')) and nparts > 1 :	# IMAGE DISPLAY
				ifu_cmd = f'ifudis -i "{fullpath(parts[1])}" -l'	# DEFAULT IS TO PLOT TRACES
				do_cmd (ifu_cmd+extn)

			elif cmd.startswith('ex') and nparts > 1 :		# EXTRACT
				ifu_cmd = 'ifuext'
				if cfg['pyfu_cfg'] is not None :
					ifu_cmd += f' --yaml {cfg["pyfu_cfg"]}'
				if cfg['trace_cfg'] is not None :
					ifu_cmd += f' --trace {cfg["trace_cfg"]}'
				if nparts == 3 :
					cfg['extr_img'] = fullpath(parts[2])
				else :
					cfg['extr_img'] = fullpath('reduced.fits')
				ifu_cmd += f' --outfiles {cfg["extr_img"]}'
				if cfg['show'] :
					ifu_cmd += ' --plot'
				do_cmd (ifu_cmd+extn)
				print ('\textracted file is',cfg['extr_img'])

			elif cmd.startswith('fh') :						# FITS HEADER
				"""
				Command:    fheader {path of FITS file}
				Function:   reads FITS header of file
				"""
				if is_fits_file (fullpath(parts[1])) :
					try :
						s = fullpath(parts[1])
						hdus = fits.open (s)
						hdr = hdus[0].header
						for key,val in hdr.items() :
							print (f'{key} : {val}')
						hdus.close()
					except :
						print ('Cannot open file',s)

			elif cmd.startswith('fl') and nparts > 1 :	# FLAT
				"""
				Command:	flat {name or directory} {optional path of masterflat}
				Function:	sets/creates a masterflat
				"""
				infile = parts[1].lower()
				if infile == 'none' :				# a) USED TO AVOID FLAT CORRECTION, or
					cfg['masterflat'] = None
				elif is_fits_file (infile) :		# b) FITS-FILE -> SET masterflat TO THIS FILE, or
					cfg['masterflat'] = fullpath(parts[1])
				else :								# c) DIRECTORY -> CREATE MASTERBIAS
					ifu_cmd = 'ifucal'
					if cfg['pyfu_cfg'] is not None :
						ifu_cmd += f' --yaml {cfg["pyfu_cfg"]}'
					ifu_cmd += f' --flat_files "{fullpath(parts[1])}"'
					if cfg['masterbias'] is not None :
						ifu_cmd += f' --subtract_bias --masterbias {cfg["masterbias"]}'
					if cfg['unitdark'] is not None :
						ifu_cmd += f' --subtract_dark --unitdark {cfg["unitdark"]}'
					if nparts == 3 :
						cfg['masterflat'] = fullpath(parts[2])
					else :
						cfg['masterflat'] = fullpath('reduced.fits')
					ifu_cmd += f' --masterflat {cfg["masterflat"]}'
					if cfg['show'] :
						ifu_cmd += ' --plot'
					cfg['calib_img'] = cfg['masterflat']
					do_cmd (ifu_cmd+extn)
				print ('\tnew masterflat file is',cfg['masterflat'])

			elif cmd.startswith('pi') and nparts > 1 :		# PIMAGE
				ifu_cmd = 'ifuima --xcol "pixel"'
				if pyfu_cfg is not None :
					ifu_cmd += f' --yaml {pyfu_cfg}'
				ifu_cmd += f' --infiles "{fullpath(parts[1])}"'
				do_cmd (ifu_cmd+extn)

			elif cmd.startswith('pl') and nparts > 1 :		# PLOT
				ifu_cmd = f'ifudis -t "{fullpath(parts[1])}"'
				do_cmd (ifu_cmd+extn)

			elif cmd.startswith('se') and nparts == 3 :		# SET ...
				if parts[1].startswith('lo') :				# ... logging
					if parts[2] == 'debug' :
						cfg['logging']['level'] = logging.DEBUG
					else :
						cfg['logging']['level'] = logging.INFO
					initialize_logging (config=cfg)
				elif parts[1].startswith('sh') :			# ... show
					if parts[2] == '1' or parts[2].lower() == 'true' or parts[2].lower() == 'on' :
						cfg['show'] = True
					else :
						cfg['show'] = False
				elif parts[1] in ['masterbias','unitdark','masterflat','pyfu_cfg','trace_cfg','working'] :
					cfg[parts[1]] = fullpath(parts[2])

			elif cmd.startswith('st') :						# STATUS
				cmd0 = 'status'
				status ()

			elif cmd.startswith('td') and nparts > 1 :		# TABLE DISPLAY
				ifu_cmd = f'ifudis -t "{fullpath(parts[1])}"'
				if nparts == 3 : # ADD SUB-TABLE NUMBER
					pass
				do_cmd (ifu_cmd+extn)

			elif cmd.startswith('tr') and nparts >= 2 :		# TRACE img new_trace_cfg
				infile = parts[1].lower()
				if is_fits_file (infile) :
					ifu_cmd = 'ifutra'
					if cfg['pyfu_cfg'] is not None :
						ifu_cmd += f' --yaml {cfg["pyfu_cfg"]}'
					ifu_cmd += f' --infile "{fullpath(parts[1])}"'
					if nparts == 3 :
						trace_cfg = fullpath(parts[2])
					else :
						trace_cfg = fullpath('trace.yaml')
					ifu_cmd += f' --save {trace_cfg}'
					if cfg['show'] :
						ifu_cmd += ' --plot'
					do_cmd (ifu_cmd+extn)
					cfg['trace_cfg'] = trace_cfg
					cfg['trace_img'] = infile
				else :
					print (infile,'is not a FITS file?')

			elif cmd.startswith('wi') and nparts > 1 :		# WIMAGE table
				ifu_cmd = 'ifuima --xcol "wavelength"'
				if pyfu_cfg is not None :
					ifu_cmd += f' --yaml {pyfu_cfg}'
				ifu_cmd += f' --infiles "{fullpath(parts[1])}"'
				do_cmd (ifu_cmd+extn)

			elif nparts == 5 and parts[1] == '=' :	# MATHS : outfile = infile1 {operation} infile2
				outfiles,nix,infiles1,func,infiles2 = parts
				ifu_cmd = 'ifucal'
				pyfu_cfg = cfg['pyfu_cfg']
				if pyfu_cfg is not None :
					ifu_cmd += f' --yaml {pyfu_cfg}'
				ifu_cmd += f' --outfiles {outfiles}'
				ifu_cmd += f' --infiles {infiles1}'
				if func == '+' :
					ifu_cmd += ' --plus'
				elif func == '-' :
					ifu_cmd += ' --minus'
				elif func == '*' :
					ifu_cmd += ' --times'
				elif func == '/' :
					ifu_cmd += ' --divided_by'
				elif func == '^' :
					ifu_cmd += ' --raised_by'
				ifu_cmd += f' --other {infiles2}'
				if cfg['show'] :
					ifu_cmd += ' --plot'
				do_cmd (ifu_cmd+extn)

			elif nparts == 4 and parts[1] == '=' :	# MATHS : outfile = {function} infile
				outfiles,nix,func,infiles = parts
				ifu_cmd = 'ifucal'
				pyfu_cfg = cfg['pyfu_cfg']
				if pyfu_cfg is not None :
					ifu_cmd += f' --yaml {pyfu_cfg}'
				if cfg['show'] :
					ifu_cmd += ' --plot'
				ifu_cmd += f' --outfiles "{outfiles}"'
				if func == 'sqrt' :
					ifu_cmd += ' --sqrt'
				elif func == 'sum' :
					ifu_cmd += ' --sum'
				elif func == 'mean' :
					ifu_cmd += ' --average'
				elif func == 'xmean' :
					ifu_cmd += ' --xmean'
				elif func == 'ymean' :
					ifu_cmd += ' --ymean'
				ifu_cmd += f' --infiles {infiles}'
				do_cmd (ifu_cmd+extn)

			elif cmd not in commands :
				print ('Not a command / command error:',cmd)
				cmd = '?'

			if normal_cmd and cmd != '' :
				cmds.append(cmd0)
				if cmd != '' :    transID += 1
				if transID > 999 : transID = 1

	except KeyboardInterrupt :
		ok = False
	save_tool_cfg ()
	sys.exit(0)

if __name__ == '__main__' :
	tool()
