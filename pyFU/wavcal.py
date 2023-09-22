#!/usr/bin/env python3

# pyfu/wavcal.py

import bisect
import logging
import numpy as np
import yaml

from astropy.io import fits
from astropy.table import Table
from scipy import signal, optimize
from matplotlib import pyplot as plt

from numpy.polynomial.polynomial import polyfit, polyval
from numpy.polynomial.legendre import legfit, legval
from numpy.polynomial.laguerre import lagfit, lagval
from numpy.polynomial.hermite import hermfit, hermval

from pyFU.display import show_with_menu
from pyFU.utils import (
    centroid,
    cubic,
    read_tables,
    line,
    vectorize,
    write_tables,
    Gaussian1D,
    get_infiles_and_outfiles,
)

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)

MIN_WAVELENGTH = 300.0  # nm
MAX_WAVELENGTH = 1000.0  # nm


def pixel2wave(pix, c, model="linear"):
    """
    Inverse function for a given pixel(wave) model.

    Converts a pixel position pix into a wavelength w using one of several invertable models:

    'linear','quadratic','cubic'	w(pix) = invert_polynomial(pix,c) because pix(w) = c[0]+c[1]*w+...
    'exp'							w(pix) = -ln[(pix-c2)/c0]/p1 =(c0/c1)+(-1)*ln[x-c2]
    'power'							w(pix) = (pix/c0)**(1/c1)
    'ln'							w(pix) = exp((pix-c0)/c1)

    The inversion is relevant because pixel(lambda) is the natural fitting
    routine (wavelengths are known a priori, not pixels).
    """
    if model in ["linear", "quadratic", "cubic"]:
        return invert_polynomial(pix, c)
    elif model == "exp":
        return -np.log((pix - c[2]) / c[0]) / c[1]
    elif model == "power":
        return (pix / c[0]) ** (1 / c[1])
    elif model == "ln":
        return np.exp((pix - c[0]) / c[1])
    else:
        raise NotImplementedError("model {0} not implemented".format(model))


def wave2pixel(w, c, model="linear"):
    """
        Converts a wavelength w into a pixel position x using one of several invertable models:

                'linear','quadratic','cubic'	x(w) = c0+c1*w+c2*w**2+c3*w**3
                'exp'							x(w) = c0*exp(-c1*w)+c2
                'power'							x(w) = c0*w**c1
                'ln'							x(w) = c0+c1*ln(w)

        The inversion is relevant because pixel(lambda) is the natural fitting routine
        (wavelengths are known), not lambda(pixel).  The 'ln' model corresponds to a constant
    velocity dispersion dln(w)/dx = const.
    """
    if model in ["linear", "quadratic", "cubic"]:
        x = w - w
        ww = 1.0
        for cc in c:
            x += cc * ww
            ww *= w
        return x
    elif model == "exp":
        return c[0] * np.exp(-c[1] * w) + c[2]
    elif model == "power":
        return c[0] * w ** c[1]
    elif model == "ln":
        return c[0] + c[1] * np.log(w)
    else:
        raise NotImplementedException("model {0} not implemented".format(model))


def invert_polynomial(pix, c):
    """
    Inverts the polynomial     pix(w) = c0+c1*w+c2*w**2+c3*w**3     to get w(pix)
    """
    nc = len(c)
    # LINE
    if nc == 2:  # x = c0+c1*w
        return (pix - c[0]) / c[1]
    # QUADRATIC
    elif nc == 3:  # x = c0+c1*w+c2*w**2
        return invert_quadratic(pix, c)
    # CUBIC
    elif nc == 4:
        return invert_cubic(pix, c)
    # ERROR
    else:
        raise NotImplementedError("polynomials with nc={0} not supported".format(nc))


def invert_quadratic(pix, p):
    return vectorize(_invert_quadratic, pix, p)


def _invert_quadratic(pix, p):
    """
    Model is 'quadratic' :
            pix(w) = p[0]+p[1]*w+p[2]*w**2 = a*w**2+b*w+c
    so find the root of the function
            f(x) = a*w**2+b*w+(c-pix) = 0
    """
    c, b, a = p
    c -= pix
    w1 = np.nan
    w2 = np.nan
    if a == 0.0:
        w1 = -c / b
    else:
        eps = 1.0
        if b < 0.0:
            eps = -1.0
        dis = b * b - 4.0 * a * c
        if dis < 0.0:
            return np.nan
        term = -b - eps * np.sqrt(dis) / a
        if term == 0.0:
            w1 = 0.0
            w2 = 0.5 * (-b + eps * np.sqrt(dis)) / a
        else:
            w1 = 0.5 * (-b - eps * np.sqrt(dis)) / a
            w2 = 2.0 * c / (-b - eps * np.sqrt(dis))
    if not np.isnan(w1) and w1 >= MIN_WAVELENGTH and w1 <= MAX_WAVELENGTH:
        return w1
    elif not np.isnan(w2) and w2 >= MIN_WAVELENGTH and w2 <= MAX_WAVELENGTH:
        return w2
    else:
        return np.nan


def invert_cubic(pix, p):
    return vectorize(_invert_cubic, pix, p)


def _invert_cubic(pix, p):
    """
    Model is 'cubic' :
            pix(w) = p[0]+p[1]*w+p[2]*w**2+p[3]*w**3 = a*w^3+b*w^2+c*w+d
    so find the root of the function
            f(x) = a*w^3+b*w^2+c*w+(d-pix) = 0
    """
    d, c, b, a = p
    d -= pix
    w1 = np.nan
    w2 = np.nan
    w3 = np.nan
    third = 1.0 / 3.0
    if a == 0.0:
        return _invert_quadratic(x, b, c, d)
    w1 = np.nan
    w2 = np.nan
    w3 = np.nan
    pp = (3.0 * a * c - b * b) / (3.0 * a * a)
    qq = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a)
    delta = qq * qq / 4.0 + pp * pp * pp / 27.0  # DISCRIMINANT

    if delta == 0.0:  # MULTIPLE SOLUTIONS
        if qq == 0.0:
            w1 = 0.0
        elif q >= 0.0:
            t1 = (0.5 * qq) ** third
            t2 = -2.0 * t1
            w1 = t1 - third * b / a
            w2 = t2 - third * b / a

    elif delta > 0.0:  # ONE REAL SOLUTION
        au = -0.5 * qq + np.sqrt(delta)
        u1 = np.abs(au) ** third
        if au < 0.0:
            u1 *= -1.0
        av = -0.5 * qq - np.sqrt(delta)
        v1 = np.abs(av) ** third
        if av < 0.0:
            v1 *= -1.0
        t1 = u1 + v1
        w1 = t1 - third * b / a

    else:  # 3 REAL SOLUTIONS
        absp = np.abs(pp)
        arg0 = -0.5 * qq * (3.0 / absp) ** 1.5
        arg1 = third * np.pi / 3.0
        arg2 = arg1 + 2 * np.pi / 3.0
        arg3 = arg1 + 4.0 * np.pi / 3.0
        f = 2.0 * np.sqrt(third * absp)

        t1 = f * np.cos(arg1)
        t2 = f * np.cos(arg2)
        t3 = f * np.cos(arg3)
        w1 = t1 - third * b / a
        w2 = t2 - third * b / a
        w3 = t3 - third * b / a

    if not np.isnan(w1) and w1 >= MIN_WAVELENGTH and w1 <= MAX_WAVELENGTH:
        return w1
    elif not np.isnan(w2) and w2 >= MIN_WAVELENGTH and w2 <= MAX_WAVELENGTH:
        return w2
    elif not np.isnan(w3) and w3 >= MIN_WAVELENGTH and w3 <= MAX_WAVELENGTH:
        return w3
    else:
        return np.nan


def cc_calibrate(
    wav,
    flx,
    refwav,
    refflx,
    show=True,
    spacing=100,
    width=100,
    model="linear",
    flux_order=None,
    flux_func="laguerre",
    wave_ranges=None,
    fix_mu=False,
):
    """
    Computes a wavelength calibration for a spectrum defined by arrays 'wav' and
    'flx'.  The wavelength table "wav" is assumed to be a rough wavelength scale
    good enough for setting the cross-correlation region, i.e. does not have to be
    very accurate.

    Returns the wavelength coefficients, the covariance matrix of those coefficients,
    the red.chi^2, the R.M.S., and the flux-calibration function of pixel-position only.

    If "flux_order" is not None, then the wavelength chunks are also used to perform a
    rough flux correction relative to the reference spectrum using a polynomial fit of
    that order.
    """
    # ---- FOR EACH CHUNK OF DATA SPECTRUM

    pixels = []  # MEAN DATA PIXEL OF EACH CHUNK
    waves = []  # MEAN WAVELENGTH OF EACH CHUNK
    err_pixels = []  # PIXEL ERRORS DERIVED FROM C-C FUNCTION
    fcorr = []  # FLUX CORRECTION FACTORS
    err_fcorr = []  # FLUX CORRECTION FACTORS
    widths = []

    if flux_func is None:
        pass
    else:
        flxfnc = flux_func.lower()
        if flxfnc.startswith("poly"):
            fit_func = polyfit
            fit_val = polyval
        elif flxfnc.startswith("lag"):
            fit_func = lagfit
            fit_val = lagval
        elif flxfnc.startswith("leg"):
            fit_func = legfit
            fit_val = legval
        elif flxfnc.startswith("her"):
            fit_func = hermfit
            fit_val = hermval
        else:
            fit_func = polyfit
            fit_val = polyval
            logging.info("using normal polynomials for flux calibration")

    nwav = len(wav)
    pix = np.arange(nwav)
    if wave_ranges is None:
        wave_ranges = [[0.0, np.inf]]

    width += width % 2  # WANT ODD NUMBER OF PIXELS IN CHUNK
    n1 = 0
    n2 = n1 + width - 1

    showing = show
    # FOR ALL WAVELENGTH CHUNKS...
    while n2 < nwav:
        w = wav[n1 : n2 + 1]
        n = n2 - n1 + 1
        x = np.arange(n)
        hanning = signal.windows.hann(n)  # signal.hanning (n)
        pixavg = 0.5 * (n1 + n2)
        wavg = np.mean(w)

        # IN RANGE OF ACCEPTABLE WAVELENGTHS?
        ok = False
        for wav_range in wave_ranges:
            if (wavg > wav_range[0]) and (wavg < wav_range[1]):
                ok = True

        if ok:
            disp = (wav[n2] - wav[n1]) / (n - 1)
            title = "chunk {0}-{1} ({2:.1f})".format(n1, n2, wavg)

            # GET CHUNK OF SPECTRUM
            s = flx[n1 : n2 + 1]
            scoef, scov = optimize.curve_fit(line, w, s)  # FIT LINE TO DATA IN WINDOW
            sfit = line(w, scoef[0], scoef[1])
            snorm = np.nanmean(s)

            # GET CORRESPONDING CHUNK OF REFERENCE
            r = np.interp(w, refwav, refflx)
            rcoef, rcov = optimize.curve_fit(line, w, r)  # FIT TO DATA IN REFERENCE
            rfit = line(w, rcoef[0], rcoef[1])
            rnorm = np.nanmean(r)
            corr = rnorm / snorm

            # PLOT CHUNKS
            f1 = (s - sfit) / snorm
            f2 = (r - rfit) / rnorm
            fscal = np.max(np.abs(f1))
            if showing:
                fig = plt.figure()
                plt.style.use("ggplot")
                plt.title(title)
                plt.xlabel("wavelength [nm]")
                plt.plot(w, f1 * hanning, color="gray", label="data*window")
                plt.plot(w, f2, color="red", label="ref")
                plt.plot(w, f1, color="black", label="data")
                plt.legend()
                plt.tight_layout()
                stat = show_with_menu(fig, ["RESULTS", "ABORT"])
                if stat == "RESULTS":
                    showing = False
                elif stat == "ABORT":
                    return None, None, None, None, None

            # CROSS-CORRELATE CHUNKS
            ycorr = np.correlate(f1 * hanning, f2, mode="same")
            yscal = np.max(np.abs(ycorr))
            xcorr = x - n // 2
            mask = signal.tukey(len(ycorr)) * np.abs(np.max(ycorr))
            tukey = mask / np.max(mask)
            if showing:
                fig = plt.figure()
                plt.style.use("ggplot")
                plt.title("C-C of " + title)
                plt.xlabel("shift in pixels")
                plt.plot(xcorr, ycorr * tukey, color="gray", label="C-C*window")
                plt.plot(xcorr, ycorr, color="black", label="C-C")

            ycorr *= tukey

            # FIND PEAK OF C-C FUNCTION
            peak, wid = centroid(xcorr, ycorr, 9)
            errpeak = 1.0
            errfcorr = 0.1 * corr
            try:
                # FIT GAUSSIAN TO C-C FUNCTION: ycorr ~ a+b*exp(-(xcorr-c)**2/d**2)
                if fix_mu:
                    # Gaussian1D with fixed mu (µ), since the latter is known by CC-argmax (lmeerwart, 10th May '23)
                    peak = xcorr[np.argmax(ycorr)]
                    def gauss_fit(x, off, ampl, sigma):
                        return Gaussian1D(x, off, ampl, peak, sigma)

                    p0 = [0.05 * peak, np.max(ycorr), wid / 2.0]
                    p, cov = optimize.curve_fit(gauss_fit, xcorr, ycorr, p0=p0)
                    wid = p[2]
                    if showing:
                        plt.plot(
                            xcorr, gauss_fit(xcorr, *p), "-.", label="fit", color="blue"
                        )
                else:
                    p0 = [0.05 * peak, np.max(ycorr), peak, wid / 2.0]
                    p, cov = optimize.curve_fit(Gaussian1D, xcorr, ycorr, p0=p0)
                    rms2 = np.nansum((ycorr - Gaussian1D(xcorr, *p)) ** 2) / len(xcorr)
                    peak = p[2]
                    wid = p[3]
                    errpeak = np.sqrt(cov[2][2] + 0.01 * wid ** 2)
                    rrfcorr = corr * np.sqrt(cov[1][1] + rms2) / snorm
                    if showing:
                        plt.plot(
                            xcorr, Gaussian1D(xcorr, *p), "-.", label="fit", color="blue"
                        )
            except:
                peak = np.nan

            if showing:
                plt.legend()
                plt.tight_layout()
                stat = show_with_menu(fig, ["RESULTS", "ABORT"])
                if stat == "RESULTS":
                    showing = False
                elif stat == "ABORT":
                    return None, None, None, None, None

            if np.isnan(peak):
                logging.warning(
                    "chunk {0:.2f} at ~{1:.2f} has bad peak".format(pixavg, wavg)
                )

            elif peak > -n / 3 and peak < n / 3:  # USE MIDDLE OF C-C FUNCTION!
                logging.debug(
                    "chunk {0:.2f} at ~{1:.2f} produced x_peak={2:.2f},dw_peak={3:.4f}".format(
                        pixavg, wavg, peak, peak * disp
                    )
                )

                # SAVE CALIBRATION POINT: AVG PIXEL WHERE WAVELENGTHS MATCH IS OFF BY THE PEAK SHIFT
                pixels.append(pixavg + peak)
                waves.append(wavg)
                err_pixels.append(errpeak)
                widths.append(wid)

                # SAVE FLUX CORRECTION FACTOR
                fcorr.append(corr)
                err_fcorr.append(errfcorr)
            else:
                logging.warning(
                    "chunk {0} produced peak too near edge {1}".format(pixavg, peak)
                )

        # NEXT CHUNK
        n1 += spacing
        n2 += spacing
        if n2 >= nwav:
            n2 = nwav - 1
        if (n2 - n1) < width // 2:
            n2 = nwav

    # FIT FLUX CORRECTION FACTORS
    cflux = None
    if flux_func is not None and flux_order is not None:
        if flux_order < 1:
            logging.error("flux-correction order < 1: {0}".format(flux_order))
        else:
            fcorr /= np.nanmedian(fcorr)
            cflux = fit_func(pixels, fcorr, flux_order, w=1.0 / np.array(err_fcorr))
            if show:
                plt.style.use("ggplot")
                plt.tight_layout()
                plt.xlabel("pixel")
                plt.ylabel("flux correction")
                plt.plot(pix, fit_val(pix, cflux), "-", color="green")
                plt.errorbar(pixels, fcorr, yerr=err_fcorr, fmt="o", color="red")
                plt.ylim(bottom=0.0)
                plt.show()
            flx *= fit_val(pix, cflux)

    # RETURN RESULTING WAVELENGTH CALIBRATION AS PACKETS OF (coef,cov,rch2,rms)
    px2wv = dispersion_fit(pixels, err_pixels, waves, model=model, show=show)
    wv2pix = dispersion_fit(pixels, err_pixels, waves, model=model, reversed=True)
    return px2wv, wv2pix, lambda x: fit_val(x, cflux)


def dispersion_fit(px, pxerr, w, model="linear", show=False, reversed=False):
    """
    Fits x(w) (if "reversed" is False) or w(x) (if "reversed" is True) to the data
    using the given model.
    Returns coefficients,covariance_matrix,red.chi^2, and R.M.S.
    """
    # ---- FIT pixel(wavelength)
    x = np.array(w)
    y = np.array(px)
    err = np.array(pxerr)
    wgt = np.median(err) / err
    n = len(x)
    xname = "wav"
    yname = "pix"
    f = wave2pixel
    label = "# No.   Wavelength Pixel     Err(Pixel) Fit(Pixel)  Diff      Diff/Err"
    datafmt = "# {0:3d}  {1:8.3f}   {2:8.3f} {3:7.3f}     {4:8.3f}   {5:8.3f}  {6:8.3f}"

    # ---- FIT wavelength(pixel)
    if reversed:
        x = np.array(px)
        y = np.array(w)
        err = x - x + 1.0e-4 * np.mean(y)  # FAKE ERROR
        xname = "pix"
        yname = "wav"
        label = (
            '# No.   Pixel     Wavelength  "Err"(Wave)  Fit(Wave)  Diff      Diff/Err'
        )
        datafmt = "# {0:3d}  {1:8.3f}  {2:8.3f}    {3:7.3f}      {4:8.3f}   {5:8.3f}  {6:8.3f}"

        # FOR NON-GENERIC MODELS, DON'T INVERT, JUST GET DIFFERENT WEIGHTING
        if model == "exp" or model == "ln":
            f = pixel2wave

    # ---- SET UP MODEL
    meanx = np.mean(x)
    if model == "linear":
        x2y = lambda xx, c0, c1: f(xx, [c0, c1], model=model)
        title = "{0}({1})=c0+c1*{1}".format(yname, xname)
        c1 = (y[-1] - y[0]) / (x[-1] - x[0])
        c0 = y[0] - c1 * x[0]
        p0 = [c0, c1]
    elif model == "quadratic":
        x2y = lambda xx, c0, c1, c2: f(xx, [c0, c1, c2], model=model)
        title = "{0}({1})=c0+c1*{1}+c2*{1}^2".format(yname, xname)
        c1 = (y[-1] - y[0]) / (x[-1] - x[0])
        c0 = y[0] - c1 * x[0]
        p0 = [c0, c1, 0.01 * c1 / meanx]
    elif model == "cubic":
        x2y = lambda xx, c0, c1, c2, c3: f(xx, [c0, c1, c2, c3], model=model)
        title = "{0}({1})=c0+c1*{1}+c2*{1}^2+c3*{1}^3".format(yname, xname)
        c1 = (y[-1] - y[0]) / (x[-1] - x[0])
        c0 = y[0] - c1 * x[0]
        p0 = [c0, c1, 0.01 * c1 / meanx, -0.0001 * c1**2 / meanx**2]
    elif model == "exp":
        x2y = lambda xx, c0, c1, c2: f(xx, [c0, c1, c2], model=model)
        if reversed:
            title = "wav(pix)=ln(c0/(wav-c2))/c1"
        else:
            title = "pix(wav)=c0*exp(-c1*wav)+c2"
        p0 = None
    elif model == "power":
        x2y = lambda xx, c0, c1: f(xx, [c0, c1], model=model)
        title = "{0}({1})=c0*{1}^c1".format(yname, xname)
        c1 = np.log(y[-1] / y[0]) / np.log(x[-1] / x[0])
        c0 = y[0] / x[0] ** c1
        p0 = [c0, c1]
    elif model == "ln":
        x2y = lambda xx, c0, c1: f(xx, [c0, c1], model=model)
        if reversed:
            title = "wav(pix)=exp((pix-c0)/c1)"
        else:
            title = "pixel(w)=p0+p1*ln(w)"
        c1 = (px[-1] - px[0]) / (np.log(w[-1]) - np.log(w[0]))
        c0 = np.log(w[0]) - c1 * px[0]
        p0 = [c0, c1]

    # ---- FIT
    coef, cov = optimize.curve_fit(x2y, x, y, sigma=err, p0=p0)

    # ---- RE-FIT FOR ROBUSTNESS
    mask = np.ones(n, dtype=bool)
    for iter in range(3):
        rms = 0.0
        for i in range(n):
            if mask[i]:
                yfit = x2y(x[i], *coef)
                d = y[i] - yfit
                rms += d**2
        m = np.sum(mask)
        rms = np.sqrt(rms / m)
        if iter == 0 and reversed:
            err = rms / wgt
        logging.info("iteration #{0}: R.M.S.: {1:.2f}".format(iter, rms))
        for i in range(n):
            yfit = x2y(x[i], *coef)
            d = np.abs(y[i] - yfit) / rms
            if d > 3:
                mask[i] = False
            err[i] = np.sqrt(err[i] ** 2 + (rms * d) ** 2)
        m = np.sum(mask)
        coef, cov = optimize.curve_fit(x2y, x[mask], y[mask], sigma=err[mask])
    if m != n:
        logging.info("removed {0} data points with >3*R.M.S. deviation".format(n - m))

    logging.info("# calibration {0}".format(title))
    fmt = "#\tcoef[{0}] = {1:10.4e} +/- {2:10.4e}"
    for i in range(len(coef)):
        logging.info(fmt.format(i, coef[i], np.sqrt(cov[i][i])))
    rchi2 = 0.0
    logging.info(label)
    for i in range(n):
        if mask[i]:
            yfit = x2y(x[i], *coef)
            d = (y[i] - yfit) / err[i]
            logging.info(datafmt.format(i + 1, x[i], y[i], err[i], yfit, d * err[i], d))
            rchi2 += d**2
    rms = np.sqrt(rchi2 / m)
    rchi2 /= m - len(coef)
    logging.info("# red. chi^2 = {0:.3f}, R.M.S.={1:.3f}".format(rchi2, rms))

    # ---- DISPLAY
    if show:
        # PLOT DATA AND FIT
        plt.style.use("ggplot")
        plt.tight_layout()
        fig = plt.figure()
        fig.subplots_adjust(right=0.8)

        plt.subplot(2, 1, 1)
        if reversed:
            plt.xlabel("pixel")
            plt.ylabel("wavelength [nm]")
        else:
            plt.xlabel("wavelength [nm]")
            plt.ylabel("pixel")
        plt.plot(x, x2y(x, *coef), "-", color="green")
        plt.errorbar(x, y, yerr=err, fmt="o", color="red")
        plt.errorbar(x[mask], y[mask], yerr=err[mask], fmt="o", color="black")
        plt.title(title)

        # PLOT RESIDUALS
        plt.subplot(2, 1, 2)
        if reversed:
            plt.xlabel("pixel")
            plt.ylabel("residual [nm]")
        else:
            plt.xlabel("wavelength [nm]")
            plt.ylabel("residual [pix]")
        dyfit = y - x2y(x, *coef)
        plt.plot([x[0], x[-1]], [0.0, 0.0], "-", color="green")
        plt.errorbar(x, dyfit, yerr=err, fmt="o", color="red")
        plt.errorbar(x[mask], dyfit[mask], yerr=err[mask], fmt="o", color="black")

        # SHOW CALIBRATION IN PLOT
        fmt = "c{0}={1:10.4e}"
        dx = x[-1] - x[0]
        ymax = np.max(dyfit + err)
        dy = ymax - np.min(dyfit - err)
        yy = ymax - 0.1 * dy
        for i in range(len(coef)):
            plt.text(x[-1] + 0.05 * dx, yy, fmt.format(i, coef[i]))
            yy -= 0.1 * dy
        plt.text(x[-1] + 0.05 * dx, yy, "red. chi^2 : {0:.2f}".format(rchi2))
        yy -= 0.1 * dy
        plt.text(x[-1] + 0.05 * dx, yy, "R.M.S. : {0:.2f}".format(rms))

        plt.show()

    # RETURN RESULTS
    return coef, cov, rchi2, rms


def cc_calibrate_spectra(
    spectra,
    refwav,
    refflx,
    pixcol="pixel",
    wavcol="wavelength",
    flxcol="flux",
    approx=None,
    show=False,
    width=100,
    spacing=100,
    model="linear",
    flux_order=None,
    flux_func="laguerre",
    wave_ranges=None,
    fix_mu=False,
    prompt=False,
) -> bool:
    """
    Calibrates a list of spectra in astropy.table.Tables by cross-correlating each spectrum with a reference.
    """
    # GLOBAL RESULTS
    idxs = []
    disps = []
    chi2s = []
    rmss = []
    showed = show

    # FOR EACH SPECTRUM...
    logging.info("cc_calibrate_spectra: spacing={0}, width={1}".format(spacing, width))
    for idx in range(len(spectra)):
        logging.info("---- spectrum #{0} ----".format(idx))
        spectrum = spectra[idx]
        pix = spectrum[pixcol]
        flux = spectrum[flxcol]

        # GET APPROXIMATE WAVELENGTH SCALE
        if wavcol in spectrum.colnames:
            wav = spectrum[wavcol]
        elif approx is not None and len(approx) > 1:
            wav = 0.0
            for i in range(len(approx)):
                wav += approx[i] * pix**i
        else:
            logging.error(
                "No approximate wavelength scale available for spectrum #{0}".format(
                    idx
                )
            )
            return False

        # CALIBRATE
        px2wv, wv2px, flxcorr = cc_calibrate(
            wav,
            flux,
            refwav,
            refflx,
            show=show,
            width=width,
            spacing=spacing,
            model=model,
            flux_order=flux_order,
            flux_func=flux_func,
            wave_ranges=wave_ranges,
            fix_mu=fix_mu
        )
        pcoef, pcov, pchi2, prms = px2wv
        if pcoef is None:
            return False
        w = pixel2wave(pix, pcoef, model=model)
        spectrum[wavcol] = w
        if flux_func is not None:
            spectrum["flux_correction"] = flxcorr(pix)
        else:
            spectrum["flux_correction"] = np.zeros(len(pix)) + 1.0
        chi2s.append(pchi2)
        rmss.append(prms)

        # SAVE METADATA
        hdr = spectrum.meta
        for i in range(len(pcoef)):
            hdr["PX2WVC{0:02d}".format(i)] = (
                pcoef[i],
                "{0}-th coefficient of pixel(wave) function".format(i),
            )
        hdr["PX2WVFUN"] = (model, "pixel(wave) function")
        hdr["PX2WVRC2"] = (pchi2, "reduced chi-square of pixel(wave) calibration")

        wcoef, wcov, wchi2, wrms = wv2px
        for i in range(len(wcoef)):
            hdr["WV2PXC{0:02d}".format(i)] = (
                wcoef[i],
                "{0}-th coefficient of wave(pixel) function".format(i),
            )
        hdr["WV2PXFUN"] = (model, "wave(pixel) function")
        hdr["WV2PXRC2"] = (wchi2, "reduced chi-square of wave(pixel) calibration")

        # DISPLAY RESULTS
        if show:
            factor = np.nanmedian(flux) / np.nanmedian(refflx)
            fig = plt.figure()
            plt.style.use("ggplot")
            plt.tight_layout()
            plt.xlabel("wavelength [nm]")
            plt.ylabel("flux")
            # plt.title ('spectrum #{0}'.format(idx))
            plt.plot(refwav, refflx * factor, "-", color="blue", label="ref")
            plt.plot(w, flux, "-", color="black", label="#{0}".format(idx))
            plt.legend(
                fontsize="x-small"
            )  # bbox_to_anchor=(1.02,1), loc='upper left', ncol=1, fontsize='x-small')
            if prompt:
                plt.show()
                ans = input("(aBORT,sILENT) :").lower()
                if ans.startswith("a"):
                    return False
                elif ans.startswith("s"):
                    show = False
            else:
                reslt = show_with_menu(fig, ["no more plots", "ABORT"])
                if reslt == "no more plots":
                    show = False
                elif reslt == "ABORT":
                    return False

        if showed:  # SAVE RESULTS
            idxs.append(idx)
            dwdp = np.median(np.diff(w) / np.diff(pix))
            disps.append(dwdp)

    rchi2 = np.mean(np.array(chi2s))
    rms = np.mean(np.array(rmss))
    logging.info(
        "Mean red.chi^2,R.M.S. of all spectra: {0:.3f},{1:.3f}".format(rchi2, rms)
    )

    if showed:
        plt.style.use("ggplot")
        plt.xlabel("Spectrum index")
        plt.ylabel("Median Dispersion [nm/pixel]")
        d = np.array(disps)
        std = np.std(d)
        plt.ylim(bottom=np.min(d) - std, top=np.max(d) + std)
        plt.plot(idxs, disps, "o")
        plt.tight_layout()
        plt.show()

    return True


def transfer_wavelengths_by_index(
    refs, extracted, pixcol="pixel", wavcol="wavelength", flxcol="flux"
) -> bool:
    """
    Transfers the wavelength calibration from a list of source spectra to a set of
    similarly extracted but not wavelength calibrated spectra.
    """
    n = len(refs)
    if n != len(extracted):
        logging.critical(
            "transfer_calibration: different number of spectra: nrefs={0}, nextracted={1}".format(
                n, len(extracted)
            )
        )
        return False

    """
	refidx = {}
	for i in range(n) :
		spectrum = refs[i]
		hdr = spectrum.meta
		key = keywds['index'][0]
		if key not in hdr :
			logging.error ('no index metadata for reference spectrum #{0}'.format(i))
		else :
			refidx[hdr[key]] = i
	"""

    for i in range(n):
        ref = refs[i]
        spectrum = extracted[i]
        refcols = ref.colnames
        cols = spectrum.colnames
        hdr = spectrum.meta
        if wavcol not in refcols:
            logging.error(
                "transfer_calibration: wavelength column names not present in reference #{0}!".format(
                    i
                )
            )
            return False
        elif len(ref) != len(spectrum):
            logging.error(
                "spectrum and target #{0} of different lengths ({1}!={2})!".format(
                    i, len(ref), len(spectrum)
                )
            )
            return False
        else:
            spectrum[wavcol] = ref[wavcol] + 0.0
            hdr["comment"] = "transfered wavelength calibration from source table"
            if "flux_correction" in ref.colnames:
                if flxcol not in cols:
                    logging.error(
                        "transfer_calibration: flux column name not present in target spectrum #{0}!".format(
                            i
                        )
                    )
                    return False
                spectrum["flux_correction"] = ref["flux_correction"] + 0.0
                spectrum[flxcol] *= spectrum["flux_correction"]
                hdr["comment"] = "transfered flux correction from source table"
    return True


def main():
    import sys
    from pyFU.utils import parse_arguments, initialize_logging

    # ---- GET DEFAULTS AND PARSE COMMAND LINE
    README = """
Python script that performs a wavelength-calibration using localized cross-correlation with a standard
spectrum of similar resolution.  This requires that the spectrum to be calibrated has roughly the same
wavelength calibration so that the regions to be cross-correlated are reasonably well-defined.
	"""
    arguments = {
        "approx": {
            "path": "wavcal:",
            "default": None,
            "flg": "-a",
            "type": list,
            "help": "rough wavelength calibration w0,d0,... where wav ~ w0+d0*pix+...",
        },
        "copy": {
            "path": None,
            "default": None,
            "flg": "-c",
            "type": str,
            "help": "copy reference wavelength and flux calibration metadata to file",
        },
        "fcol": {
            "path": "wavcal:",
            "default": "flux",
            "flg": "-f",
            "type": str,
            "help": "name of flux table column in reference",
        },
        "flxcol": {
            "path": "wavcal:",
            "default": "flux",
            "flg": "-F",
            "type": str,
            "help": "name of output flux table column)",
        },
        "flux_function": {
            "path": "wavcal:",
            "default": None,
            "flg": "-X",
            "type": str,
            "help": "function for flux-calibration; polynomial|legendre|laguerre|hermite",
        },
        "flux_order": {
            "path": "wavcal:",
            "default": 5,
            "flg": "-M",
            "type": str,
            "help": "polynomial order for C-C flux-correction",
        },
        "generic": {
            "path": None,
            "default": None,
            "flg": "-G",
            "type": str,
            "help": "YAML file for generic wavelength calibration info",
        },
        "infiles": {
            "path": "wavcal:",
            "default": None,
            "flg": "-i",
            "type": str,
            "help": "pathname(s) of input FITS or ascii table(s)",
        },
        "in_format": {
            "path": "wavcal:",
            "default": None,
            "flg": "-I",
            "type": str,
            "help": "optional format for finding spectrum in the pathname directory",
        },
        "outfiles": {
            "path": "wavcal:",
            "default": None,
            "flg": "-o",
            "type": str,
            "help": "output FITS table(s)",
        },
        "out_format": {
            "path": "wavcal:",
            "default": None,
            "flg": "-O",
            "type": str,
            "help": "optional format for writing calibrated spectra to the pathname directory",
        },
        "model": {
            "path": "wavcal:",
            "default": "linear",
            "flg": "-m",
            "type": str,
            "help": "model (linear|quadratic|cubic|exp|power",
        },
        "pause": {
            "path": None,
            "default": False,
            "flg": "-P",
            "type": bool,
            "help": "pause/prompt after every spectral calibration",
        },
        "pixcol": {
            "path": "wavcal:",
            "default": "pixel",
            "flg": "-x",
            "type": str,
            "help": "name of pixel table column in target",
        },
        "plot": {
            "path": None,
            "default": False,
            "flg": "-p",
            "type": bool,
            "help": "plot result",
        },
        "reference": {
            "path": "wavcal:",
            "default": None,
            "flg": "-r",
            "type": str,
            "help": "pathname of FITS reference spectrum",
        },
        "save": {
            "path": "wavcal:",
            "default": None,
            "flg": "-Y",
            "type": int,
            "help": "YAML file for saving wavelength parameters",
        },
        "spacing": {
            "path": "wavcal:",
            "default": 120,
            "flg": "-s",
            "type": int,
            "help": "spacing of cross-correlation windows [pix]",
        },
        "wave_ranges": {
            "path": "wavcal:",
            "default": None,
            "dshow": "all",
            "flg": "-R",
            "type": list,
            "help": "wavelength ranges used l1,l2,l3,l4,...",
        },
        "wcol": {
            "path": "wavcal:",
            "default": "wavelength",
            "flg": "-w",
            "type": str,
            "help": "name of wavelength table column in reference",
        },
        "wavcol": {
            "path": "wavcal:",
            "default": "wavelength",
            "flg": "-W",
            "type": str,
            "help": "name of output wavelength table column",
        },
        "window_cc": {
            "path": "wavcal:",
            "default": 120,
            "flg": "-N",
            "type": int,
            "help": "size of cross-correlation windows [pix]",
        },
        "window_centroid": {
            "path": "wavcal:",
            "default": 11,
            "flg": "-C",
            "type": int,
            "help": "size of cross-correlation centroid window [pix]",
        },
        "fix_mu": {
            "path": "wavcal:",
            "default": False,
            "flg": "-MU",
            "type": bool,
            "help": "whether to fix mu to CC-peak and therefore only fit peak width",
        },
        "yaml": {
            "path": None,
            "default": None,
            "flg": "-y",
            "type": str,
            "help": "global YAML configuration file for parameters",
        },
    }
    args, cfg = parse_arguments(arguments)
    info = cfg["wavcal"]

    # ---- LOGGING
    initialize_logging(config=cfg)
    logging.info("*************************** wavcal ******************************")

    # ---- GET INPUT AND OUTPUT FILE NAMES
    infiles, outfiles = get_infiles_and_outfiles(args.infiles, args.outfiles, cfg=info)
    if "in_format" not in info:
        info["in_format"] = None

    # ---- OUTPUT GENERIC CONFIGURATION FILE?
    if args.generic is not None:
        logging.info(
            "Appending generic wavelength calibration configuration info to"
            + args.generic
        )
        with open(args.generic, "a") as stream:
            yaml.dump({"wavcal": info}, stream)
        sys.exit(0)

    # ---- GET REFERENCE
    if "reference" in info and info["reference"] is not None:
        logging.info('Reading reference file {info["reference"]} ...')
        reftab, hdr = read_tables(pathname=info["reference"])
        if len(reftab) < 1:
            logging.critical("reference file does not contain any spectra!")
            sys.exit(1)
        refwav = reftab[0][info["wcol"]]
        refflx = reftab[0][info["fcol"]]
    else:
        logging.critical("no reference file given!")
        sys.exit(1)

    # ---- FOR ALL TABLES
    for infile, outfile in zip(infiles, outfiles):
        spectra, header = read_tables(pathname=infile)
        if len(spectra) == 0:
            logging.critical(f"no spectra read from {infile}")

        ok = True

        # ---- TRANSFER WAVELENGTH AND FLUX CALIBRATIONS
        if args.copy:
            logging.info(
                f"transfering wavelength calibration from reference to {infile}"
            )
            if not transfer_wavelengths_by_index(
                reftab, spectra, wavcol=info["wavcol"], flxcol=info["flxcol"]
            ):
                logging.error(
                    "ABORTED spectral calibration via transfer from reference!"
                )
                ok = False

        # ---- OR PERFORM C-C CALIBRATION
        else:
            logging.info(f"C-C wavelength calibration of {infile} using reference")
            if not cc_calibrate_spectra(
                spectra,
                refwav,
                refflx,
                pixcol=info["pixcol"],
                wavcol=info["wavcol"],
                flxcol=info["flxcol"],
                approx=info["approx"],
                show=args.plot,
                width=info["window_cc"],
                spacing=info["spacing"],
                model=info["model"],
                flux_order=info["flux_order"],
                flux_func=info["flux_function"],
                wave_ranges=info["wave_ranges"],
                fix_mu=info["fix_mu"],
                prompt=cfg["pause"],
            ):
                logging.error("ABORTED wavelength calibration!")
                ok = False

        # ---- SAVE RESULTS
        if outfile is not None and ok:
            logging.info(f"Saving to FITS table file {outfile}...")
            if "out_format" not in info:
                info["out_format"] = None
            write_tables(
                spectra, header=header, pathname=outfile, fmt=info["out_format"]
            )

        logging.info(
            "*****************************************************************\n"
        )


if __name__ == "__main__":
    main()
