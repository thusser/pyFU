#!/usr/bin/env python3

# pyfu/barebones.py

"""
This is a non-functional pyFU module showing the basics of dealing with configurations
coming from the command line, a YAML file, and the internal defaults.
"""

import logging

ROOTKEY = 'barebones'	# FAKE pyFU CONFIGURATION KEYWORD

class bones (object) :
	"""
	Dummy class to show how to handle defaults.
	"""
	def __init__ (self, info) :
		self.text = 'Ciao, mondo!'	# INTERNAL DEFAULT

		# CHECK FOR ALTERNATE VAUES IN CONFIGURATION SUB-DICTIONARY info
		somethings = ['text']
		for key in somethings :
			if key in info :
				self.__dict__[key] = info[key]
				logging.info ('{0} : {1}'.format(key,self.__dict__[key]))

	def result (self) :
		logging.info (self.text)

def main () :
	import sys
	from pyFU.utils import parse_arguments, initialize_logging
	from pyFU.utils import get_infiles_and_outfiles

    # ---- GET DEFAULTS AND PARSE COMMAND LINE
	README = """
This is a non-functional pyFU module showing the basics of dealing with configurations
coming from the command line, a YAML file, and any defaults.
	"""
	arguments = {
		'blahblah':{'path':ROOTKEY+':text','default':'Hello, world!', \
			'flg':'-B','type':str, 'help':f'meaningless text entry in the {ROOTKEY} configuration'},
		'generic':{'path':None,'default':None, \
			'flg':'-G','type':str,'help':'YAML file for generic calibration info'},
		'infiles':{'path':ROOTKEY+':','default':None, \
			'flg':'-i','type':str,'help':'pathname(s) of ignored input FITS or ascii table(s)'},
		'nix':{'path':'nix','default':'NIX', \
			'flg':'-N','type':str, 'help':'a nothing value at the topmost configuration level'},
		'outfiles':{'path':ROOTKEY+':','default':None, \
			'flg':'-o','type':str, 'help':'pathname(s) of ignored output FITS table(s)'},
		'yaml':{'path':None,'default':None, \
			'flg':'-y','type':str,'help':'pathname of global YAML configuration file for parameters'}
		}
	args,cfg = parse_arguments (arguments, readme=README)	# , verbose=True)
	if ROOTKEY in cfg :
		info = cfg[ROOTKEY]
	else :
		info = cfg
	print (cfg)

	# ---- LOGGING
	initialize_logging (config=cfg)
	logging.info (f'*************************** {ROOTKEY} ******************************')

	# ---- OUTPUT GENERIC CONFIGURATION FILE?
	if args.generic is not None :
		logging.info (f'Appending generic {ROOTKEY} configuration info to {args.generic}')
		with open (args.generic,'a') as stream :
			yaml.dump ({ROOTKEY:info}, stream)
		sys.exit(0)

	# ---- DO SOMETHING WITH A COMMAND LINE PARAMETER
	logging.info (f'{args.nix} = {cfg["nix"]}')

	# ---- GET INPUT AND OUTPUT FILE NAMES SO THAT WE COULD ACTUALLY DO SOMETHING
	infiles,outfiles = get_infiles_and_outfiles (args.infiles,args.outfiles,cfg=info)

	# ---- GET OBJECT THAT PARSES INTERNAL DEFAULT VALUES AND EXTERNAL CONFIGURATIONS
	knochen = bones(info)
	knochen.result()

	logging.info ('*****************************************************************\n')

if __name__ == '__main__' :
	main ()
