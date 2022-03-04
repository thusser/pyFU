# mgauss.py

"""
Multiple Gauss-function fits.

The multiple-Gauss function used in trace.py is specialized for IFU use, so here is a
general-purpose fitter that also takes care of fitting thick things with multiple
components.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.signal             import find_peaks, peak_widths, peak_prominences
from astropy.modeling.models  import Gaussian1D, Const1D, Polynomial1D
from astropy.modeling.fitting import LevMarLSQFitter

from pyFU.utils import smooth, multiple_gauss_function

class MultipleGaussians (object) :
	def __init__ (self) :
		self._model  = None
		self._ipeaks = None
		self._wpeaks = None
		self._apeaks = None
		self._fit    = None

	def find_peaks (self, x, y, max_peaks=5) :
		"""
		Find main peaks in y.  The default max_peaks=5 is simply so that pyFU's standard ampl_format
		can list 16 coefficients (3*5+1) as a single hex number.
		"""
		ipeaks = list(np.array(find_peaks(y)[0],dtype=int))
		wpeaks = list(np.array(peak_widths (y, ipeaks, rel_height=np.exp(-0.5))[0]/2,dtype=int))	# SIGMAS
		apeaks = list(peak_prominences (y, ipeaks)[0])
		"""
		print (ipeaks)
		print (wpeaks)
		print ('#, ipeak, xpeak, xwid-, xwid+')
		for i in range(len(ipeaks)) :
			print (i,ipeaks[i],x[ipeaks[i]],x[int(ipeaks[i]-wpeaks[i])],x[int(ipeaks[i]+wpeaks[i])])
		print (apeaks)
		"""

		# Separate really wide peak into multiple peaks
		imin = np.argmin(wpeaks)
		imax = np.argmax(wpeaks)
		n = (wpeaks[imax]//wpeaks[imin])//2
		if n > 1 :
			# print (imin,imax,n)
			ix = ipeaks[imax]
			iw = (wpeaks[imax]*2)//n
			ipeaks[imax] = ix-iw*n//2
			wpeaks[imax] = iw
			apeaks[imax] = 0.8*apeaks[imax]
			for i in range(1,n) :
				ipeaks.append(ix-iw*n//2+i*iw)
				wpeaks.append(iw)
				apeaks.append(apeaks[imax])
			"""
			print ('extended peaks:')
			print (ipeaks)
			print (wpeaks)
			print (apeaks)
			"""

		# SORT PEAKS BY AMPLITUDE
		opeak = np.argsort (apeaks)
		npeaks = min(max_peaks,len(ipeaks))
		self._ipeaks = []
		self._wpeaks = []
		self._apeaks = []
		for i in range(npeaks) :
			idx = opeak[i]
			self._ipeaks.append(ipeaks[idx])
			self._wpeaks.append(wpeaks[idx])
			self._apeaks.append(apeaks[idx])

	def define_model (self, x, y) :
		""" Construct the model to be fitted with starting parameters. """
		self._model = Const1D(np.min(y))	# Polynomial1D (1)
		for i in range(len(self._ipeaks)) :
			amp = self._apeaks[i]
			pos = x[self._ipeaks[i]]
			iw = int(i-self._wpeaks[i])
			if iw < 0 :
				iw = int(i+self._wpeaks[i])
			sig = np.abs(pos-x[iw])/2	# ????
			self._model += Gaussian1D (amp,pos,sig)

	def apply_fit (self, x) :
		""" Applies fit to some data. """
		return self._fit(x)

	def fit_model (self, x, y) :
		"""
		Fit the data with a non-linear least-squares fitter.
		The result is an object that can apply the fit to data.
		"""
		fitter = LevMarLSQFitter()
		self._fit = fitter (self._model, x, y)

	def plot_components (self, x) :
		for model in self._fit :
			plt.plot (x, model(x), '--', color='lightgray', label=None)

	def analyze (self, x, y, plot=False) :
		s = smooth(y)
		self.find_peaks (x,s)
		self.define_model (x,y)
		self.fit_model (x,y)
		# pars = self.parameters ()
		yfit = self.apply_fit(x)
		resid = (y-yfit)**2
		chi2 = np.sum(resid)/np.std(resid)

		# Plot the data and the best fit
		if plot :
			self.plot_components (x)
			plt.plot(x, s, 'o', color='blue', label='smoothed')
			plt.plot(x, y, 'o', color='black', label='original')
			plt.plot(x, y-yfit, 'o', color='red', label='diff')
			# plt.plot(x, multiple_gauss_function(x,*pars), '.', color='green', label='function')
			plt.plot(x, y*0.,'--',color='black', label=None)

			xx = np.linspace(x[0],x[-1],1000)
			plt.plot(xx, self.apply_fit(xx),'-',color='red')
			m = len(self._ipeaks)
			plt.title (f'const+{m}xGaussians: chi^2={chi2:.2f}')
			plt.legend ()
			plt.show ()

	def parameters (self) :
		""" Returns a list containing all coefficients of the model. """
		if self._fit is None :
			return self._model.parameters
		else :
			return self._fit.parameters

if __name__ == '__main__' :
	"""
	Test method using dumps of amplitude traces from ifutra (see option "save to file" after
	amplitude fit plot).
	"""
	from astropy.table import Table
	tab = Table.read ('trace_amplitudes.csv',format='ascii.csv',header_start=0,data_start=1)
	x = tab['x']
	y = tab['amp']
	mg = MultipleGaussians ()
	mg.analyze (x,y,plot=True)
