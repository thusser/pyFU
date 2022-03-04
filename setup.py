#!python
from setuptools import setup, find_packages

setup(
	name = 'pyFU',
	version = '0.2',
	description = 'simple IFU spectral image reduction software',
	author = 'Frederic V. Hessman',
	author_email = 'hessman@astro.physik.uni-goettingen.de',
	requires = ['argparse','astropy','bisect','matplotlib','numpy','parse','scipy','skimage','yaml'],
	entry_points = {'console_scripts': [
		'ifucal = pyFU.calib.main',
		'ifudis = pyFU.display.main',
		'ifuext = pyFU.extract.main',
		'ifufak = pyFU.fake.main',
		'ifuima = pyFU.image.main',
		'ifulam = pyFU.lamp.main',
		'ifureb = pyFU.rebin.main',
		'ifusol = pyFU.solar.main',
		'ifutra = pyFU.trace.main',
		'ifuwav = pyFU.wavcal.main'
		]}
	)
