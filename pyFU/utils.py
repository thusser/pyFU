# pyFU/utils.py

import argparse
import bisect
import datetime
import logging
import numpy as np
import os
import parse
import sys
import yaml

from astropy.io import fits
from astropy.table import Table, Column
from matplotlib import pyplot as plt
from scipy import signal, optimize, integrate
from scipy.ndimage import gaussian_filter1d

from pyFU.defaults import (
    pyFU_default_formats,
    pyFU_logging_level,
    pyFU_logging_format,
    pyFU_logging_file,
)
from pyFU.meta import HeaderFilter

PYFU_EXTENSIONS = {
    ".FIT": "fits",
    ".fit": "fits",
    ".fits": "fits",
    ".xml": "votable",
    ".vot": "votable",
    ".dat": "ascii",
    ".txt": "ascii",
    ".csv": "ascii.csv",
}


def const_func(x, a):  # 0-TH ORDER, 1 COEF
    return a + x - x


def linear_func(x, a, b):  # 1-ST ORDER, 2 COEF
    return a + b * x


def quadratic_func(x, a, b, c):  # 2-ND ORDER, 3 COEF
    return a + b * x + c * x**2


def cubic_func(x, a, b, c, d):  # 3-RD ORDER, 4 COEF
    return a + b * x + c * x**2 + d * x**3


def quartic_func(x, a, b, d, c, e):  # 4-TH ORDER, 5 COEF
    return a + b * x + c * x**2 + d * x**3 + e * x**4


def quintic_func(x, a, b, d, c, e, f):  # 5-TH ORDER, 6 COEF
    return a + b * x + c * x**2 + d * x**3 + e * x**4 + f * x**5


polynomial_functions = [
    const_func,
    linear_func,
    quadratic_func,
    cubic_func,
    quartic_func,
    quintic_func,
    None,
    None,
    None,
    None,
]


def check_directories(filename, create=True):
    """
    Check to see if the directories of the given filename exist.
    If not and "create=True", then they are created.
    """
    dname = os.path.dirname(filename)
    if dname == "":
        return True
    if not os.path.isdir(dname):
        if create:
            try:
                os.makedirs(dname)
            except FileExistsError:
                logging.error(
                    "os.path.isdir does not see " + dname + " but os.makedirs does?"
                )
                return False
        else:
            return False
    return True


def csv2list(s):
    """
    Takes a string of comma-separated values and returns a list of str's, int's, or float's.
    """
    l = s.split(",")  # SPLIT CONTENTS OF STRING
    try:
        if "." in s:  # PROBABLY CONTAINS float
            for i in range(len(l)):
                l[i] = float(l[i])
        else:  # PROBABLY CONTAINS int
            for i in range(len(l)):
                l[i] = int(l[i])
    except:  # OR GO BACK TO SIMPLE STRINGS
        l = s.split(",")
    return l


def construct_new_path(pattern, filename, prefix="new_"):
    """
    Using a simple pattern, modify a filename to construct a full new path by adding a
    prefix or a suffix to the basename, or by extracting the inner content (full path to
    the left, file extension on the right) and replacing a "*" with that content. If the
    pattern doesn't include an asterix ("*"), then the default prefix is used.
    """
    pathname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    parts, extension = os.path.splitext(filename)
    kernl = basename.replace(extension, "")

    if pattern is None:
        return pathname + "/" + prefix + basename
    elif pattern.endswith("*"):
        return pathname + "/" + pattern.replace("*", "") + basename
    elif pattern.startswith("*"):
        return pathname + "/" + basename + pattern[1:]
    elif "*" in pattern:
        return pattern.replace("*", kernl)
    else:
        return pathname + "/" + prefix + basename


def convolve_rebin_table(
    tab,
    dlambda=None,
    wave_out=None,
    model="interpolate",
    pixcol="pixel",
    wavcol="wavelength",
    flxcol="flux",
    errcol="err_flux",
    other=None,
):
    """
    Convolves/rebins the entries of a spectral table (like convolve_rebin() but with more Table support).
    The "other" list contains names of numerical data columns that should also be convolved/rebinned;
    without "other", rebinning will result in the loss of table columns whose interpolated.
    Returns a new table.
    """
    assert wavcol is not None
    assert flxcol is not None
    t = Table(tab)
    for col in [wavcol, flxcol, errcol]:
        if (col is not None) and (col not in t.colnames):
            logging.error("{0} not a table column label".format(col))
            return None

    # PRESENT MEDIAN DISPERSION
    dispersion = np.median(np.diff(t[wavcol]))
    logging.debug(f"mean dispersion: {str(dispersion)}")

    # CONVOLVE?
    if dlambda is not None:
        resol = dlambda / dispersion  # WAVELENGTH -> PIXELS
        cols = [flxcol, errcol]
        if other is not None:
            cols += other
        for col in cols:
            if col is not None:
                logging.debug(f"Convolving column {col} with Gaussian...")
                t[col] = gaussian_filter1d(t[col], resol)

    # REBIN?
    if wave_out is not None:
        outtab = Table()
        outtab.meta = tab.meta
        err_in = None
        if errcol is not None:
            err_in = t[errcol]
        if model == "histogram":
            logging.debug("Histogram rebinning...")
            outtab[flxcol], err = hist_rebin(t[wavcol], t[flxcol], err_in, wave_out)
            outtab[flxcol].units = "nm"
            if errcol is not None:
                outtab[errcol] = err
                outtab[errcol].units = "nm"
        elif model == "interpolate":
            logging.debug("Interpolating...")
            f = np.interp(wave_out, t[wavcol], t[flxcol])
            outtab[flxcol] = Column(
                f, unit=t[flxcol].unit, description="interpolated flux"
            )
            if errcol is not None:
                f = np.interp(wave_out, t[wavcol], t[errcol])
                outtab[errcol] = Column(
                    f, unit=t[flxcol].unit, description="interpolated flux error"
                )
        if (pixcol is not None) and (pixcol in t.colnames):
            logging.debug("Interpolating...")
            p = np.interp(wave_out, t[wavcol], t[pixcol])
            outtab[pixcol] = Column(p, unit="pix", description="interpolated pixel")
        if other is not None:
            for col in other:
                if col is not None:
                    logging.debug(f"Interpolating column {col} ...")
                    x = np.interp(wave_out, wave_in, t[col])
                    outtab[col] = Column(
                        x, unit=t[col].unit, description="interpolated"
                    )
    else:
        outtab = t

    # RETURN NEW TABLE
    return outtab


def convolve_rebin(
    wave_in=None,
    flux_in=None,
    err_in=None,
    dlambda=None,
    wave_out=None,
    model="interpolate",
):
    """
    Convolves the input spectrum to a given dlambda (as opposed to a given R=lambda/dlambda!) and
    performs a rebinning to a new wavelength sampling.
    Returns one or two arrays.
    """
    if wave_in is None and flux_in is None:
        raise ValueError("No input data given!")

    # PRESENT MEDIAN DISPERSION
    dispersion = np.median(np.diff(wave_in))

    # CONVOLVE TO TARGET WAVELENGTH RESOLUTION
    err = err_in
    if dlambda is not None:
        resol = dlambda / dispersion  # WAVELENGTH -> PIXELS
        conv = gaussian_filter1d(flux_in, resol)
        if err_in is not None:
            err = gaussian_filter1d(err_in, resol)
    else:
        conv = flux_in
    err_out = err

    # REBIN TO THE INPUT WAVELENGTH ARRAY
    if wave_out is not None:
        if model == "histogram":
            bconv, err_out = hist_rebin(wave_in, conv, err, wave_out)
        elif model == "interpolate":
            bconv = np.interp(wave_out, wave_in, conv)
            if err_in is not None:
                err_out = np.interp(wave_out, wave_in, err_in)
        else:
            raise NotImplementedError(
                "{0} is not an implemented convolve model ".format(model)
            )
        if err_out is not None:
            return bconv, err_out
        else:
            return bconv
    else:
        if err_out is not None:
            return conv, err
        else:
            return conv


def fits_to_np(x, y, hdr, range=False):
    """
    Conversion of 2-D FITS int pixel coordinates to numpy array coordinates.
    The "range" option assumes that x and y contain initial and final
    coordinates, e.g. x=[1,3] implies the numpy range [0,3], and that the
    output is j1,j2,i1,i2 (y --> j, x --> i).

    Example :
            NAXIS2=10, x=[1,3], y=[5,7] --> j=[10-5,10-7], i=[1-1,3-1] ==> 5,4,0,3
    """
    ny = hdr["NAXIS2"]
    if range:
        return [ny - y[0], ny - y[1] + 1, x[0] - 1, x[1]]
    else:
        return [ny - y, x - 1]


def get_sec(hdr, key="BIASSEC"):
    """
    Returns the numpy range for a FITS section based on a FITS header entry using the standard format

            {key} = '[{col1}:{col2},{row1}:row2}]'

    where 1 <= col <= NAXIS1, 1 <= row <= NAXIS2.
    """
    if key in hdr:
        s = hdr.get(key)  # WITHOUT CARD COMMENT
        ny = hdr["NAXIS2"]
        sx = s[s.index("[") + 1 : s.index(",")].split(":")
        sy = s[s.index(",") + 1 : s.index("]")].split(":")
        return [ny - int(sy[1]), ny - int(sy[0]) + 1, int(sx[0]) - 1, int(sx[1])]
    else:
        return None


def get_list_of_paths_and_filenames(path_pattern, mode="both"):
    """
    Returns a list of paths, filenames, or both using the given pathname and pattern.
    The pattern should include a "*" if it refers to multiple files,
    e.g. "/mydir/x*.fits" would enable getting a list like

            [("/mydir/x01.fits","x01.fits"),("/mydir/x02.fits","x0002.fits"), ...]
    """
    if path_pattern is None:
        raise ValueError("get_list_of_paths_and_filenames: no path_pattern given")

    files = []
    things = []

    # EXTRACT DIRECTORY NAME AND FILE PATTERN
    pathname = os.path.dirname(path_pattern)
    if pathname == "" or pathname == None:
        pathname = "./"
    pattern = os.path.basename(path_pattern)

    # GET LIST OF ALL FILES IN THAT DIRECTORY
    for dirpath, dirnames, filenames in os.walk(pathname):
        things += filenames
        break  # ONLY GO ONE PATH LEVEL
    if len(things) == 0:
        logging.error("no input files!")
        return files

    # SEPARATE PATTERN INTO BEGINNING AND ENDING PARTS
    if pattern is None:
        parts = None
    else:
        parts = pattern.split("*")

    # FOR ALL FILES, SEE IF THEY MATCH THE PATTERN
    for f in things:
        if parts is None:
            name = pathname + "/" + f
            if mode == "path":
                files.append(name)
            elif mode == "name":
                files.append(f)
            else:
                files.append((name, f))
        elif len(parts) == 1:
            if f == pattern:
                name = pathname + "/" + f
                if mode == "path":
                    files.append(name)
                elif mode == "name":
                    files.append(f)
                else:
                    files.append((name, f))
        else:
            prefix = parts[0] == "" or f.startswith(parts[0])
            suffix = parts[1] == "" or f.endswith(parts[1])
            if prefix and suffix:
                name = pathname + "/" + f
                if mode == "path":
                    files.append(name)
                elif mode == "name":
                    files.append(f)
                else:
                    files.append((name, f))

    # RETURN LIST OF FULL PATHS AND FILENAMES THAT MATCH
    return files


def get_infiles_and_outfiles(
    infiles: str, outfiles: str, cfg=None, overwrite=False
) -> (list, list):
    """
    Intelligently construct lists of input and output files.
    The accepted formats for "infiles" and "outfiles" are
            1. a single filename
            2. a comma-separated list of filenames
            3. an implicit list of filenames using a "*" as a wildcard, e.g. "/mydir/myfiles_*.fits"
                    (must be framed by '"' to avoid pre-parsing by the shell!)
    If "infiles" contains "*", then the individual content of "*" in an input file is transferred
    to the corresponding output file, e.g. infiles="in_*.fits" and outfiles="out_*.fits" implies
    that "in_0001.fits" will be transformed to "out_0001.fits".

    if "infiles" or "outfiles" is None, then look for these keywords in the configuration
    dictionary "cfg".
    """
    inlist, outlist = None, None

    # NO INPUT GIVEN - MAYBE IN THE CONFIGURATION FILE?
    if infiles is None:
        if cfg is not None:
            if "infiles" in cfg:
                infiles = cfg["infiles"]
            elif "infile" in cfg:
                infiles = cfg["infile"]
    if outfiles is None:
        if cfg is not None:
            if "outfiles" in cfg:
                outfiles = cfg["outfiles"]
            elif "outfile" in cfg:
                outfiles = cfg["outfile"]
    if infiles is None and outfiles is None:
        return ([], [])

    # COMMA-SEPARATED LISTS
    if infiles is None:
        inlist = [None]
    else:
        if "," in infiles:
            inlist = infiles.split(",")
        else:
            inlist = [infiles]
    if outfiles is None:
        outlist = [None]
    else:
        if "," in outfiles:
            outlist = outfiles.split(",")
        else:
            outlist = [outfiles]

    # IMPLICIT LIST OF INPUT AND/OR OUTPUT FILES
    if infiles is not None and "*" in infiles:
        iprefix = infiles[
            : infiles.index("*")
        ]  # e.g. infiles='in*.fits' --> iprefix=infiles[0:2]='in'
        isuffix = infiles[
            infiles.index("*") + 1 :
        ]  #                             oprefix=infiles[3:]='.fits'
        inlist = get_list_of_paths_and_filenames(infiles, mode="path")

        if outfiles is not None and "*" in outfiles:
            outlist = []
            if "*" in outfiles:
                oprefix = outfiles[: outfiles.index("*")]
                osuffix = outfiles[outfiles.index("*") + 1 :]
            else:
                oprefix = iprefix
                osuffix = oprefix
            for infile in inlist:
                if infile.startswith(iprefix) and infile.endswith(isuffix):
                    l = len(infile)
                    kernl = infile[
                        len(iprefix) : l - len(isuffix)
                    ]  # e.g. infile='in0001.fits' --> infile[2:11-5]=infile[2:6]='0001'
                    outlist.append(oprefix + kernl + osuffix)
                else:
                    logging.warning(
                        f"{infile} changed after get_list_... : former prefix={iprefix} and suffix={isuffix}"
                    )

    # FINAL CHECK
    if inlist is None or outlist is None:  # SOMETHING WENT WRONG - NO MATCHES FOR BOTH
        logging.warning("could not construct input and/or output file lists")
        return ([], [])
    elif not overwrite and len(inlist) == len(outlist):
        for inf, outf in zip(inlist, outlist):
            if inf is not None and inf == outf and not overwrite:
                raise ValueError(f"{inf}={outf} but no overwrite permission!")
    return (inlist, outlist)


def Gaussian1D(x, a, b, c, d):
    """
    Simple 1-D Gaussian function
    """
    return a + b * np.exp(-((x - c) ** 2) / d**2)


def initialize_logging(
    level=pyFU_logging_level,
    form=pyFU_logging_format,
    logfile=pyFU_logging_file,
    config=None,
    config_file=None,
):
    """
    Configures the global logging environment, either from a YAML
    file, a dictionary, or from the given arguments.
    """
    cfg = config
    if config_file is not None:
        with open(config_file) as stream:
            cfg = yaml.safe_load(stream)
    usefile = logfile
    if cfg is not None:
        if "logging" in cfg:
            cfg = cfg["logging"]
        if "level" in cfg and cfg["level"] is not None:
            level = cfg["level"]
        if "format" in cfg and cfg["format"] is not None:
            form = cfg["format"]
        if "file" in cfg and cfg["file"] is not None:
            logfile = cfg["file"]
        if "file" in cfg:
            logfile = cfg["file"]
    logging.basicConfig(level=level, format=form)
    if logfile is not None:
        logger = logging.getLogger("app")
        stream = open(logfile, "a")
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(pyFU_logging_format)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)


def is_number(s):
    """Checks to see if the string is a number."""
    if "." in s:
        try:
            n = float(s)
        except:
            return False
    else:
        try:
            n = int(s)
        except:
            return False
    return True


def list2csv(l):
    """Converts a list to a string of comma-separated values."""
    s = None
    if isinstance(l, list):
        s = str(l[0])
        for i in range(1, len(l)):
            s += "," + str(l[i])
    return s


def load_csv_table(filename, labels=None):
    """
    Create a Table from a simple comma-separated text file.
    If the column labels aren't passed but are given in a single comment line, they are used.
    """
    tab = None
    if not labels is None:  # COLUMNS KNOWN
        tab = Table(names=labels)
    with open(filename, "r") as f:
        for l in f:
            if l.startswith("#"):
                if labels is None:  # GET LABELS FROM COMMENT LINE?
                    things = l[1:].split(",")
                    tab = Table(names=things)
            else:
                things = l.split(",")
                stuff = []
                for thing in things:
                    try:
                        if not "." in thing:
                            stuff.append(int(thing))
                        else:
                            stuff.append(float(thing))
                    except:
                        stuff.append(thing.strip())
                tab.add_row(stuff)
    return tab


def merge_dictionaries(dict1, dict2):
    """
    Merge the secondary dictionary dict2 into the prime dictionary dict1 if there's something new to merge.
    """
    if dict2 is None:
        return
    assert isinstance(dict1, dict)
    assert isinstance(dict2, dict)

    # FOR EVERYTHING IN SECONDARY DICTIONARY...
    for key2, val2 in dict2.items():

        # IF NOT PRESENT, SIMPLY ADD TO PRIMARY
        if key2 not in dict1 or dict1[key2] is None:
            try:
                dict1[key2] = val2
            except TypeError as e:
                logging.error(
                    str(e) + "\nkey2={0}\nval2={1}\ndict1={2}".format(key2, val2, dict1)
                )
                raise TypeError("problem in merge_dictionaries")
        # IF PRESENT AND A DICTIONARY, USE A RECURSIVE MERGE
        elif isinstance(val2, dict):
            merge_dictionaries(dict1[key2], val2)

    return True


def multiple_gauss_function(x, *c):
    """
    Handy function when referencing the astropy.model fit results from mgauss.py

    Coefficients for n components are either
            amp1,pos1,sig1,amp2,pos2,sig2,...			(no background: N=0+3*n)
    or
            const,pos1,sig1,amp2,pos2,sig2,...			(constant background: N=1+3*n)
    or
            const,slope,pos1,sig1,amp2,pos2,sig2,...	(linear background: N=2+3*n)
    """
    n = len(c)
    m = n // 3  # NUMBER OF GAUSSIAN   PARAMETERS
    l = n - m * 3  # NUMBER OF BACKGROUND PARAMETERS
    if l == 0:
        mgf = x - x
    elif l == 1:
        mgf = x - x + c[0]
    else:
        mgf = c[0] + c[1] * x
    for i in range(m):
        ii = l + i * 3
        mgf += c[ii] * np.exp(-0.5 * (x - c[ii + 1]) ** 2 / c[ii + 2] ** 2)
    return mgf


def hist_integral(x, y, yerr, wx, x_1, x_2, normalize=False, strict=False):
    """
    \int_x1^x2 y dx for previously binned data (like CCD pixels).
    x are the positions of the x-bin centres.
    wx is the array of widths of the x-bins, where the bins are assumed
    to touch, i.e.  x[i]+0.5*wx[i] = x[i+1]-0.5*wx[i+1].
    """
    # TRIVIAL VALIDITY TEST
    if strict and (x1 < x[0] or x2 <= x[0] or x1 >= x[-1] or x2 > x[-1]):
        return np.nan, np.nan

    x1 = x_1 + 0.0
    x2 = x_2 + 0.0
    if x1 < x[0]:
        x1 = x[0]
    if x2 > x[-1]:
        x2 = x[-1]
    if wx is None:
        wx = np.concatenate((np.diff(x), [x[-1] - x[-2]]))  # SAME SIZE AS x

    n = len(x)
    yint = 0.0
    if yerr is None:
        errors = False
        err = None
    else:
        errors = True
        err = 0.0

    # GET INDEX RANGE
    i1 = np.searchsorted(x, x1)  # INDEX OF DATA x-VALUE >= x1
    i2 = np.searchsorted(x, x2)

    # EVERYTHING WITHIN A SINGLE BIN
    if i1 == i2:
        ilow = 0
        ihi = 0
        xmid = x[i2] - 0.5 * wx[i2]  # MID-POINT

        if x1 < xmid:  # LOWER CONTRIBUTION
            z1 = (min(xmid, x2) - x1) / wx[i2 - 1]
            dyint = y[i2 - 1] * max(0.0, z1) * wx[i2 - 1]
            if errors:
                err += yerr[i2 - 1] ** 2 * max(0.0, z1) ** 2
            yint += dyint

        if x2 > xmid:  # UPPER CONTRIBUTION
            z2 = (x2 - max(xmid, x1)) / wx[i2]
            dyint = y[i2] * max(0.0, z2) * wx[i2]
            if errors:
                err += yerr[i2] ** 2 * max(0.0, z2) ** 2
            yint += dyint

    # SPREAD OVER LOWER FRACTIONAL BIN, MIDDLE BINS, AND UPPER FRACTIONAL BIN
    else:

        # LOWER FRACTION
        xmid = x[i1] + 0.5 * wx[i1]

        if x1 < xmid:  # FULL CONTRIBUTION FROM UPPER FRACTIONAL BIN
            z1 = (xmid - x1) / wx[i1]
            dyint = y[i1] * max(0.0, z1) * wx[i1]
            if errors:
                err += yerr[i1] ** 2 * max(0.0, z1) ** 2
            yint += dyint

            ilow = i1 + 1

        else:  # NO CONTRIBUTION FROM LOWER FRACTIONAL BIN
            z2 = (x[i1 + 1] - x1) / wx[i1 + 1]
            dyint = y[i1 + 1] * max(0.0, z2) * wx[i1 + 1]
            if errors:
                err += yerr[i1 + 1] ** 2 * max(0.0, z2) ** 2
            yint += dyint

            ilow = i1 + 2

        # UPPER FRACTION
        xmid = x[i2] - 0.5 * wx[i2]

        if x2 < xmid:  # NO CONTRIBUTION FROM UPPER FRACTIONAL BIN
            z1 = (x2 - (x[i2 - 1] - 0.5 * wx[i2 - 1])) / wx[i2 - 1]
            dyint = y[i2 - 1] * max(0.0, z1) * wx[i2 - 1]
            if errors:
                err += yerr[i2 - 1] ** 2 * max(0.0, z1) ** 2
            yint += dyint

            ihi = i2 - 2

        else:  # FULL CONTRIBUTION FROM LOWER FRACTIONAL BIN
            z2 = (x2 - xmid) / wx[i2]
            dyint = y[i2] * max(0.0, z2) * wx[i2]
            if errors:
                err += yerr[i2] ** 2 * max(0.0, z2) ** 2
            yint += dyint

            ihi = i2 - 1

        # MIDDLE FRACTION
        dyint = 0.0
        for i in range(ilow, ihi + 1, 1):
            dyint += y[i] * wx[i]
            if errors:
                err += yerr[i2] ** 2
        yint += dyint

    if errors:
        err = np.sqrt(err)
    if normalize:
        yint /= x2 - x1
        if errors:
            err /= x2 - x1
    return yint, err


def hist_rebin(x, y, yerr, xref, debug=False, normalize=False):
    """
    Rebins the (x,y,err_y) data in a table to match the x-scale of xref.
    The x-point are assumed to represent the centers of spectral bins,
    so the rebinning includes the integration of each pixel from one side to the other.
    If yerr is not None, the binned errors are also computed and returned.

    Returns y_rebin,err_y_rebin
    """
    nx = len(x)
    nref = len(xref)
    ybin = xref - xref
    if yerr is not None:
        err_ybin = ybin - ybin
    else:
        err_ybin = None

    # GET RIGHT-HANDED TOTAL BIN SIZES
    dx = np.concatenate((np.diff(x), [x[-1] - x[-2]]))  # SAME SIZE AS x

    # FOR ALL xref BINS
    dxref = xref[1] - xref[0]  # STARTING BIN
    for i in range(nref):
        x1ref = xref[i] - 0.5 * dxref
        x2ref = x1ref + dxref
        y_int, err_int = hist_integral(
            x, y, yerr, dx, x1ref, x2ref, normalize=normalize
        )
        ybin[i] = y_int
        if yerr is not None:
            err_ybin[i] = err_int

        if i < nref - 1:  # GET NEXT BIN
            dxref = 0.5 * (dxref + xref[i + 1] - xref[i])

    # RETURN RESULT
    return ybin, err_ybin


def write_tables(
    tables,
    pathname="./",
    header=None,
    overwrite=True,
    fmt=None,
    keywords=None,
    formats=None,
):
    """
    Writes a list of Tables to some supported formats.
    If the format string is not given, then "pathname" is the file name.
    If the format string is given, the spectra are written to multiple
    files, e.g. 'spectrum_{0:00d}.txt', and the pathname is the directory used.
    The FITS "header" is used when writing to a FITS binary table file: the
    header is then included in the primary HDU.
    If "keywords" is not None, then index-based pyFU keywords are not copied
    from the header.
    """
    if fmt is not None:
        name, ftype = os.path.splitext(fmt)
    else:
        name, ftype = os.path.splitext(pathname)
    if ftype not in PYFU_EXTENSIONS:
        logging.error(
            "unknown file format for {0} or missing prefix format string: {1}".format(
                pathname, ftype
            )
        )
        return

    filter = HeaderFilter(header=header, keywords=keywords)

    if ftype == ".fit" or ".fits":
        # SINGLE FITS FILE
        if fmt is None:
            phdu = fits.PrimaryHDU()
            hs = [phdu]
            if header is not None:
                filter.copy_header(header, phdu.header)
            for i in range(len(tables)):
                t = tables[i]
                thdu = fits.table_to_hdu(t)
                hs.append(thdu)
                logging.debug(
                    "appending spectrum #{0} to HDU list of {1}...".format(
                        i + 1, pathname
                    )
                )

            hdus = fits.HDUList(hs)
            hdus.writeto(pathname, overwrite=overwrite)

        # MULTIPLE FITS FILES
        else:
            for i in range(len(tables)):
                t = tables[i]
                name = pathname
                if not pathname.endswith("/"):
                    name += "/"
                name += fmt.format(i)
                if hdu is not None:
                    filter.copy_header(hdu.header, t.meta, bare=True)
                logging.debug("writing spectrum #{0} to {1}...".format(i + 1, name))
                t.write(name, format="fits", overwrite=overwrite)

    else:
        # SINGLE TEXT FILE
        if fmt is None:
            tab = Table()
            for i in range(len(tables)):
                t = tables[i]
                for key in t.colnames():
                    newkey = key + "__{0:00d}".format(i)
                tab[newkey] = t[key]
                logging.debug(
                    "appending spectrum #{0} to {1}...".format(i + 1, pathname)
                )
                t.write(pathname, format=PYFU_EXTENSIONS[ftype], overwrite=overwrite)

        # MULTIPLE TEXT FILES
        else:
            for i in range(len(tables)):
                t = tables[i]
                name = pathname
                if not pathname.endswith("/"):
                    name += "/"
                name += fmt.format(i)
                logging.info("writing spectrum #{0} to {1}...".format(i + 1, name))
                t.write(name, format=PYFU_EXTENSIONS[ftype], overwrite=overwrite)


def find_peaks(yarr, w=5, pad=None, positive=True):
    """
    Returns a list of peaks in an array found by fitting parabolas to the data within
    a window of "w" elements.  Any peaks within "pad" of the edges are ignored.
    If "positive" is False, then any peak found is returned, otherwise only positive
    peaks are returned (default).
    """
    narr = len(yarr)
    xarr = np.arange(narr)
    if pad is None:
        pad = w
    dx = pad // 2
    peaks = []
    npeaks = []
    for i in range(pad, narr - pad):
        x1, x2 = xarr[i - dx], xarr[i + dx]
        xp, pars = parabolic_peak(xarr, yarr, i, w=5)
        if not np.isnan(xp):
            peaks.append(xp)
    peaks = np.sort(peaks)
    results = []
    numbers = []
    ipeaks = []
    k = 0
    pold = None
    for peak in peaks:
        ipeak = int(peak)
        # NOT IN RESULTS
        if not (ipeak in ipeaks):
            # TOO CLOSE TO PREVIOUS ENTRY
            if (pold is not None) and (np.abs(peak - pold) < w):
                # print ('close',peak,results,numbers,k,pold)
                results[k - 1] += peak
                numbers[k - 1] += 1
            # NEW VALUE: ADD TO RESULTS
            else:
                # print ('new',peak,results,numbers,k,pold)
                results.append(peak)
                numbers.append(1)
                ipeaks.append(ipeak)
                k += 1
                pold = ipeak
        # ALREADY IN RESULTS
        else:
            # print ('old',peak,results,numbers,k,pold)
            results[k - 1] += peak
            numbers[k - 1] += 1
    # print (results,numbers)
    return np.array(results) / np.array(numbers)


def parabolic_peak(x, y, x0, w=5, positive=True):
    """
    Fits a parabola to a local region of a dataset (x,y).
    Returns the center of the parabola and the polynomial constants.
    """
    # GET REGION
    n = len(x)
    x1 = x0 - w // 2
    x2 = x0 + w // 2
    i1 = int(x1 + 0.5)
    i2 = int(x2 + 0.5)
    if i2 >= n:
        i2 = n - 1
        i1 = i2 - w + 1
    if i1 < 0:
        i1 = 0
        i2 = i1 + w - 1
    xx = x[i1 : i2 + 1]
    yy = y[i1 : i2 + 1]

    coef, cov = optimize.curve_fit(quadratic_func, xx, yy)
    c, b, a = coef  # BACKWARDS!

    # CHECK THAT THE SOLUTION YIELDS A POSITIVE PEAK
    if (positive and b <= 0.0) or (a == 0.0):
        return np.nan, coef

    # CHECK THAT SOLUTION REMAINS WITHIN THE WINDOW
    xp = -0.5 * b / a
    if xp < x1 or xp > x2:
        return np.nan, coef

    return xp, coef


def show_hdu(
    hdu,
    vmin=None,
    vmax=None,
    aspect=None,
    colourbar=False,
    flip=False,
    kappa=3.0,
    fits_coords=True,
):
    """
    Display an image from a FITS HDU using pixel-centered coordinates..

    If "kappa" is given, then only a region above and below (+/-kappa*stddev) the median value is displayed.

    If "flip" is True, then the images are displayed with the numpy y-origin on top, which is the
    computer science standard; the lower left corner of the bottom left pixel is then (x,y) = (-0.5,NY-0.5)
    and the upper right corner of the upper right pixel is (NX-0.5,-0.5).

    If "fitscoord" is True, then the pixel coordinates are displayed in the FITS standard : not flipped,
    lower left corner of the lower left pixel is -0.5,-0.5 and upper right corner of the upper right pixel
    is NX+0.5,NY+0.5).
    """
    hdr = hdu.header
    xmin, xmax, ymin, ymax, zmin, zmax = get_image_limits(hdu)
    xmin = -0.5
    xmax += 0.5

    # GET COORDINATES OF EXTREME IMAGE LIMITS INCLUDING PIXEL SIZES
    if flip:
        ymin = ymax + 0.5
        ymax = -0.5
    elif fits_coords:
        flip = False
        xmin, ymin = -0.5, -0.5
        ymax += 0.5

    zmed, zsig = np.median(hdu.data), np.std(hdu.data)
    if vmax is not None:
        zmax = vmax
    elif kappa is not None:
        zmax = zmed + kappa * zsig
    if vmin is not None:
        zmin = vmin
    elif kappa is not None:
        zmin = zmed - kappa * zsig
    plt.clf()
    if flip:
        origin = "upper"
    else:
        origin = "lower"
    data = np.array(hdu.data, dtype=float) + 0.0
    im = plt.imshow(
        data,
        interpolation="none",
        aspect=aspect,
        origin=origin,
        extent=(xmin, xmax, ymin, ymax),
        vmax=zmax,
        vmin=zmin,
    )
    if colourbar:
        plt.colorbar(im)
    return im


def vector2Table(hdu, xlabel="wavelength", ylabel="flux"):
    """
    Reads a 1-D vector from a FITS HDU into a Table.
    If present, the wavelength scale is hopefully in a simple, linear WCS!
    """
    hdr = hdu.header
    if hdr["NAXIS"] != 1:
        logging.error("vector2Table can only construct 1-D tables!")
        return None
    nw = hdr["NAXIS1"]
    pixl = np.arange(nw)
    wave = None

    # GET FLUX
    bscale = 1.0
    bzero = 0.0
    """
	if 'BSCALE' in hdr and 'BZERO' in hdr :
		bscale = hdr['BSCALE']
		bzero  = hdr['BZERO']
	"""
    flux = hdu.data * bscale + bzero

    # GET WAVELENGTH
    if "CRVAL1" in hdr and "CDELT1" in hdr:  # SIMPLE WCS
        crpix1 = 1
        if "CRPIX1" in hdr:
            crpix1 = hdr["CRPIX1"]
        w0 = hdr["CRVAL1"]
        dwdx = hdr["CDELT1"]
        wave = w0 + dwdx * (pixl + 1 - (crpix1 - 1))

    # GET UNITS
    if "CUNIT1" in hdr:
        cunit1 = hdr["CUNIT1"]
    elif wave is not None:  # ASSUME ASTRONOMERS USE ANGSTROMS
        cunit1 = "nm"
        wave /= 10.0
    else:
        cunit1 = "pix"

    # CONSTRUCT Table
    t = Table()
    if wave is not None:
        t[xlabel] = Column(wave, unit=cunit1, description=xlabel)
    else:
        t[xlabel] = Column(pixl, unit=cunit1, description=xlabel)
    t[ylabel] = Column(flux, unit="unknown", description=ylabel)
    t.meta = hdr
    return t


def read_tables(pathname=None, fmt=None):
    """
    Reads spectra as a list of Tables from one or more files.
    If no format string is given, then the path name is the name of the file.
    If the format string is given, the spectra are read from multiple
    files, e.g. 'spectrum_*.txt', and the pathname is the directory used.
    returns a list of tables and the primary HDU header, if available.
    """
    tables = []
    headers = []
    header = None

    if fmt is not None:
        name, ftype = os.path.splitext(fmt)
        fullname = pathname + "/" + fmt
    else:
        name, ftype = os.path.splitext(pathname)
        fullname = pathname
    if ftype not in PYFU_EXTENSIONS:
        logging.error(
            "unknown file format for {0} or missing prefix format string: {1}".format(
                pathname, ftype
            )
        )
        return None

    files = get_list_of_paths_and_filenames(fullname)
    for f, name in files:
        if ftype == ".fit" or ftype == ".fits" or ftype == ".fits.gz":
            hdus = fits.open(f)
            header = hdus[0].header
            for i in range(1, len(hdus)):
                hdu = hdus[i]
                hdr = hdu.header
                # BINARY TABLE
                if "XTENSION" in hdr and hdr["XTENSION"] == "BINTABLE":
                    header = hdr
                    t = Table.read(hdus, hdu=i)
                    t.meta["FILENAME"] = name + "#{0}".format(i)
                    tables.append(t)
                    headers.append(header)
                # 1-D "IMAGE"
                elif "NAXIS1" in hdr:
                    t = vector2Table(hdu)
                    if t is not None:
                        t.meta["FILENAME"] = name
                        tables.append(t)
                        headers.append(header)

        # READ CONGLOMERATED TABLE
        elif fmt is None:
            logging.info("reading conglomerated ascii spectra {0} ...".format(f))
            try:
                t = Table.read(f, format=PYFU_EXTENSIONS[ftype])
                # SEPARATE INTO INDIVIDUAL TABLES
                tabs = {}
                for key in t.colnames():
                    if "__" in key:
                        try:
                            i = key.rindex("_")
                            idx = int(key[i + 1 :])
                            oldkey = key[: i - 1]
                            if not idx in tabs:
                                tabs[idx] = Table()
                                tabs.meta["FILENAME"] = +name + "#{0}".format(i)
                                tables.append(tabs[idx])
                            tabs[idx][oldkey] = t[key]
                        except ValueError:
                            pass
            except Exception as e:
                logging.info(str(e))

        # MULTIPLE TEXT FILES
        else:
            logging.info("reading ascii spectrum {0} ...".format(f))
            t = Table.read(pathname + "/" + f, format=extensions[ftype])
            if "FILENAME" not in t.meta:
                t.meta["FILENAME"] = f
            tables.append(t)
            headers.append(header)

    # RETURN RESULT
    return tables, headers


def get_image_limits(hdu, mode="number"):
    """
    Get size and intensity limits from an image stored in a FITS HDU (python coordinates!).
    """
    hdr = hdu.header
    data = hdu.data
    xmin = 0
    xmax = hdr["NAXIS1"] - 1
    ymin = 0
    ymax = hdr["NAXIS2"] - 1
    zmin = np.nanmin(data)
    zmax = np.nanmax(data)

    if mode == "outside":  # INCLUDE SIZE OF PIXELS, E.G. FOR pyplot.imshow()
        xmin -= 0.5
        xmax += 0.5
        ymin -= 0.5
        ymax += 0.5

    return xmin, xmax, ymin, ymax, zmin, zmax


def centroid1D(yarr, pos, width, get_sigma=False, get_fwhm=False, subt_bkg=True):
    """
    Get the centroid of 1-D x and y sub-arrays at a particular position and window width.
    """
    w = int(width - 1) // 2
    i1 = int(pos - w)
    if i1 < 0:
        i1 = 0
    i2 = int(i1 + width - 1)
    if i2 >= len(yarr):
        i2 = len(yarr) - 1
    i1 = int(i2 - width + 1)

    n = len(yarr)
    xarr = np.arange(n)
    x = xarr[i1 : i2 + 1]
    y = yarr[i1 : i2 + 1]
    if subt_bkg:
        bkg = np.min(y)
    else:
        bkg = 0.0
    cntrd = np.sum(x * (y - bkg)) / np.sum(y - bkg)
    width = 3.0 * np.sqrt(
        np.abs(np.sum((y - bkg) * (x - cntrd) ** 2) / np.sum(y - bkg))
    )
    i = int(cntrd + 0.5)
    mx = yarr[i]
    i1, i2 = i - 1, i + 1
    while i1 > 0 and yarr[i1] > 0.5 * mx:
        i1 -= 1
    while i2 < n - 1 and yarr[i2] > 0.5 * mx:
        i2 += 1
    x1 = (0.5 * mx - yarr[i1] * (i1 + 1) + yarr[i1 + 1] * i1) / (
        yarr[i1 + 1] - yarr[i1]
    )
    x2 = (0.5 * mx - yarr[i2 - 1] * i2 + yarr[i2] * (i2 - 1)) / (
        yarr[i2] - yarr[i2 - 1]
    )
    fwhm = x2 - x1
    if np.abs(fwhm - (i2 - i1)) > 1:
        fwhm = i2 - i1

    if not get_sigma and not get_fwhm:
        return cntrd
    elif get_sigma and not get_fwhm:
        return cntrd, width
    elif get_fwhm and not get_sigma:
        return cntrd, fwhm
    else:
        return cntrd, width, fwhm


def peak_local_max_1D(arr, min_distance=5, threshold_abs=None, threshold_rel=None):
    """
    Simple 1-D replacement for scikit.features.peak_local_max(), which is too finicky.
    """
    if threshold_abs is not None:
        threshold = threshold_abs
    elif threshold_rel is not None:
        threshold = np.max(arr) * threshold_rel
    else:
        threshold = np.min(arr)

    n = len(arr)
    peaks = []
    for i in range(min_distance, n - min_distance):
        arri = arr[i]
        if arri > arr[i - min_distance] and arri > arr[i + min_distance]:
            if len(peaks) > 0:
                di = i - peaks[-1][0]
                if di <= min_distance and peaks[-1][1] < arri:  # LAST ONE NOT AS GOOD?
                    peaks[-1] = [i, arri]
            else:
                peaks.append([i, arri])
    return peaks[:][0]


def centroid(x, y, m, subtract_median=False, subtract_min=False):
    """
    Returns the centroid and a measure of the width of an array y(x)
    for m values around the peak.
    If "subtract_median" is True, then the median value is first subtracted.
    """
    n = len(x)
    sumxy = 0.0
    sumy = 0.0
    ysub = 0.0
    if subtract_median:
        ysub = np.median(y)
    elif subtract_min:
        ysub = np.min(y)
    peak = np.argmax(y)  # INDEX OF HIGHEST PEAK
    for i in range(peak - m // 2, peak + m // 2):
        if i >= 0 and i < n:
            sumxy += (y[i] - ysub) * x[i]
            sumy += y[i] - ysub
    if np.isnan(sumy) or sumy == 0.0:
        return np.nan, np.nan

    x0 = sumxy / sumy
    sumydx2 = 0.0
    for i in range(peak - m // 2, peak + m // 2 + 1):
        if i >= 0 and i < n:
            dx = x[i] - x0
            sumydx2 += (y[i] - ysub) * dx * dx
    w = 3.0 * np.sqrt(np.abs(sumydx2 / sumy))
    return x0, w


def read_spectrum(filename, hdu=1):
    """
    Extracts a spectrum table from a FITS or ascii table.
    """
    if filename.endswith(".csv"):
        table = Table.read(filename, format="ascii.csv")
        return table  # [wcol],table[fcol],table.meta
    elif filename.endswith(".txt") or filename.endswith(".dat"):
        table = Table.read(filename, format="ascii.tab")
        return table
    elif filename.endswith(".fits"):
        hdus = fits.open(filename)
        if len(hdus) == 1:
            table = vector2Table(hdus[0])
        else:
            table = Table.read(hdus, hdu=hdu)
        return table
    else:
        logging.error("Unable to read {0}".format(filename))
        sys.exit(1)


def write_spectra(filename, spectra, pheader, overwrite="True"):
    """
    Writes a list of spectra Tables to a FITS table file.
    "pheader" is the header of the original file.
    """
    phdu = fits.PrimaryHDU(header=pheader)
    hdus = [phdu]
    for spectrum in spectra:
        hdus.append(BinTableHDU(spectrum))
    writeto(filename, overwrite=overwrite)


def write_spectrum(tab, filename):
    """
    Writes a spectrum table to a FITS or ascii table.
    """
    if filename.endswith(".csv"):
        table.write(filename, format="ascii.csv")
    elif filename.endswith(".txt") or filename.endswith(".dat"):
        table.write(filename, format="ascii.tab")
    elif filename.endswith(".fits"):
        table.write(filename, format="fits")
    else:
        raise Exception("Unable to write {0}".format(filename))


def cubic(x, a, b, c):
    return a + b * x + c * x**2 + d * x**3


def poly(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3


def line(x, a, b):
    return a + b * x


def cubic_equation(a, b, c, d):
    """
    Solves the cubic equation
            a*x^3+b*x^2+c*x+d = 0
    by reducing to the depressed cubic
            t^3+p*t+q=0
            x = t-b/(3*a)
            p = (3*a*c-b^2)/(3*a^2)
            q = (2*b^3-9*a*b*c+27*a^2*d)/(27*a^3)
    which, using Vieta's substitution
            t = w-p/(3*w)
    becomes
            w^3+q-p**3/(27*w^3) = 0
    or the quadratic equation
            (w^3)^2+q*(w^3)-p^3/27. = 0
    which has the roots
            w1
    """
    raise NotImplementedException("cubic_equation")


def parse_arguments(arguments, readme=None, config=None, parser=None, verbose=False):
    """
    Extends argparse command line parsing with the possibility of using a default YAML dictionary.
    The input dictionary "arguments" contains

            keyword: {'path':dict_path,'default':actual_default,
                                    'flg':one_char_flag,'type':type,'help':help_text}

    dict_path is a string encoding where in the dictionary the value should be placed, e.g.

            'path':'scatter:model:polynomial:order'

    means that the argument should be placed as {'scatter':{'model':{'polynomial':{'order':HERE}}}}.
    If no path is given, then the value is placed at the highest level of the configuration dictionary.
    If the path does not end in ":", then the parameter name given is the last path entry, otherwise
    it's the parsed name.

    argparse is not given any real defaults: 'default' is the default displayed in the help text and
    the default used after the argparse arguments have been combined with a YAML dictionary so that
    the YAML can supercede the argparse defaults and the command line can supercede the YAML values.

    Returns the classic argparse argument dictionary, the updated dictionary, and the infokey
    sub-dictionary, if used.
    """
    if config is None:
        config = {}

    # ---- CREATE PARSER AND PARSE COMMAND LINE
    if parser is None:
        parser = argparse.ArgumentParser(readme)
    for arg in arguments:
        udict = arguments[arg]
        for key in ["default", "flg", "type", "help"]:
            if key not in udict:
                raise ValueError(
                    "user dictionary {0} does not contain {1}!".format(str(udict), key)
                )
        flag = "--{0}".format(arg)
        if "flg" in udict:
            flg = udict["flg"]
        else:
            flg = "-?"
        if "dshow" in udict:
            dshow = udict["dshow"]
        else:
            dshow = udict["default"]
        hlp = "{0} (default {1})".format(udict["help"], dshow)
        if udict["type"] is bool:
            if udict["default"]:
                parser.add_argument(
                    flag, flg, default=None, action="store_false", help=hlp
                )
            else:
                parser.add_argument(
                    flag, flg, default=None, action="store_true", help=hlp
                )
        elif udict["type"] is list:
            parser.add_argument(flag, flg, default=udict["default"], type=str, help=hlp)
        else:
            parser.add_argument(
                flag, flg, default=udict["default"], type=udict["type"], help=hlp
            )
    args = parser.parse_args()

    # ---- UPDATE DICTIONARY WITH YAML FILE FOR USER DEFAULTS
    if "yaml" in arguments and args.yaml is not None:
        with open(args.yaml) as stream:
            d = yaml.safe_load(stream)
        if verbose:
            print("merging", config, d)
        merge_dictionaries(config, d)  # ONLY ADDS NEW ENTRIES FROM d
        if verbose:
            print("\nparse_arguments:\n", config)


    # ---- UPDATE CONFIGURATION WITH COMMAND LINE INPUT
    adict = args.__dict__  # DICTIONARY OF ARGUMENTS FROM argparse (WITH ALL KEYS!)
    for arg in adict:  # AS IN args.{arg}
        udict = arguments[arg]
        loc = config
        key = arg
        if verbose:
            print(
                "\nparsing arg=",
                arg,
                "\n\tval=",
                adict[arg],
                "\n\tdict=",
                udict,
                "\n\tloc=",
                loc,
                "...?",
            )

        # PARSE DICTIONARY PATH TO GET FINAL DESTINATION
        if "path" in udict and udict["path"] is not None:
            levels = udict["path"].split(":")
            for level in levels[:-1]:  # FOR ALL BUT LAST PATH ENTRY
                if verbose:
                    print("\nlevel", level)
                if level not in loc:  # ADD MISSING CONFIGURATION ENTRIES
                    if verbose:
                        print("\nadding dictionary at level", level)
                    loc[level] = {}
                loc = loc[level]
                if verbose:
                    print("\nnew loc", loc, "\n\tlevel", level)
            if (
                levels[-1] != ""
            ):  # IF LAST LEVEL GIVEN IS BLANK (udict['path'] ENDS WITH ":"),
                key = levels[
                    -1
                ]  # KEY IS THAT LEVEL (OTHERWISE IT'S THE ORIGINAL ARGUMENT)
        if verbose:
            print("\nfinal level", loc, "\n\tkey", key, key in loc)
            if key in loc:
                print("\n\tcontextual value", loc[key])

        # NO argparse, YAML, OR COMMAND-LINE CONTENT: USE DEFAULT
        if key not in loc:
            content = udict["default"]
        else:
            content = loc[key]

        # CONVERT LISTS FROM str TO int OR float OR LIST OF LISTS
        if (
            (content is not None)
            and (udict["type"] is list)
            and isinstance(content, str)
        ):
            if verbose:
                print("\ncontent=", content)
            a = str(content)
            if a.startswith("["):
                l = []
                b = a.replace("[[", "[").replace("]]", "]").split("]")
                for c in b:
                    if (c is not None) and (c != ""):
                        l.append(csv2list(c[1:]))
            else:
                l = csv2list(a)
            content = l

        # TRANSFER TO DICTIONARY
        if verbose:
            print("\nadding ", content, "\n\tto", key)
        loc[key] = content


    # ---- RESULTS
    if verbose:
        print("\nparse_arguments:\n", config)
    return args, config


def smooth(x, width=5, window="hanning"):
    """Derived from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html"""
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < width:
        raise ValueError("Input vector needs to be bigger than window size.")
    if width < 3:
        return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            'Window is on of "flat", "hanning", "hamming", "bartlett", "blackman"'
        )
    s = np.r_[x[width - 1 : 0 : -1], x, x[-2 : -width - 1 : -1]]
    if window == "flat":  # moving average
        w = np.ones(width, "d")
    else:
        w = eval("np." + window + "(width)")
    y = np.convolve(w / w.sum(), s, mode="same")
    m = (len(y) - len(x)) // 2
    return y[m:-m]


def vectorize(func, x, *args):
    """
    Runs func(x,*args) on an np.array "x" when it normally wouldn't work.
    """
    if isinstance(x, float) or isinstance(x, int):
        return func(x, *args)
    elif isinstance(x, np.ndarray) or isinstance(x, list):
        n = len(x)
        f = np.zeros(n)
        for i in range(n):
            f[i] = func(x[i], *args)
        return f
    else:
        raise NotImplementedError("cannot vectorize {0}".format(str(type(x))))


def UTC_now():
    return datetime.datetime.utcnow()


if __name__ == "__main__":
    print("Testing vectorize()...")

    def nix(x, c0, c1):
        return c0 + c1 * x

    print("nix=", nix(*(1, 2, 3)))
    y = np.arange(10)
    a = 1
    b = 10
    print(vectorize(nix, y, *(a, b)))

    """
	import parse
	print ('Testing strip_format...')
	fmt = 'COE{0:03d}_{1:1d}'
	s = fmt.format(2,3)
	print (s)
	stuff = sscanf(fmt,s)
	print (stuff,len(stuff))
	"""

    import matplotlib.pyplot as plt

    x = np.arange(100)
    y = np.sin(2.0 * np.pi * x / 12.345) + 0.05 * (2 * np.random.randn(100) - 1)
    p = peak_local_max_1D(y, min_distance=4)
    print(x, y, p)
    plt.plot(x, y, "o", color="black")
    # plt.plot(x[p],y[p],'+',color='red')
    plt.show()
