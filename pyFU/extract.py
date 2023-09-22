#!/usr/bin/env python3

# pyfu/extract.py

import numpy as np
import sys
import matplotlib as mpl
import logging

from astropy.io import fits
from astropy.table import Table, Column
from matplotlib import pyplot as plt
from scipy.special import erf

from pyFU.defaults import pyFU_default_keywords, pyFU_default_formats
from pyFU.meta import HeaderFilter
from pyFU.trace import SpectrumTracer
from pyFU.utils import merge_dictionaries, get_infiles_and_outfiles

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)


class SpectrumExtractor(object):
    """
    Using the information from a SpectrumTracer, this object
    extracts and bins the individual spectra using either a
    simple integrated Gaussian weighting method or a semi-optimal extraction.

    See Sections 3.2.1 and 3.2.2 in Sandin et al. 2010, A&A 515, A35.
    """

    def __init__(self, tracer, config=None, keywords=None, formats=None):
        self.tracer = tracer

        self.keywords = keywords
        if keywords is None:
            self.keywords = dict(pyFU_default_keywords)

        self.formats = formats
        if formats is None:
            self.formats = dict(pyFU_default_formats)

        # EXTRACTION CONFIGURATION ...
        info = config
        if "extract" in config:
            info = config["extract"]

        self.ron = 0.0  # e-
        key = "ron"
        if info is not None and key in info:
            self.ron = info[key]

        self.gain = 1.0  # e-/ADU
        key = "gain"
        if info is not None and key in info:
            self.gain = info[key]

        self.pixcol = "pixel"
        key = "pixcol"
        if info is not None and key in info:
            self.pixcol = info[key]

        self.flxcol = "flux"
        key = "flxcol"
        if info is not None and key in info:
            self.flxcol = info[key]

        self.errcol = "err_flux"
        key = "errcol"
        if info is not None and key in info:
            self.errcol = info[key]

        self.labcol = "label"
        key = "labcol"
        if info is not None and key in info:
            self.labcol = info[key]

        self.idxcol = "index"
        key = "idxcol"
        if info is not None and key in info:
            self.idxcol = info[key]

        # TRACE CONFIGURATION ...
        self.sigma = 3.0  # DEFAULT JUST IN CASE
        self.number_traces = tracer.number_of_traces()
        if self.number_traces <= 0:
            logging.critical(
                "no traces available from tracer ({0})".format(self.number_traces)
            )

        # GET pyFU HEADER FILTER
        self.filter = HeaderFilter(keywords=self.keywords, formats=self.formats)

    def get_gain_ron(self, hdu, verbose=False):
        """
        Extract the gain [e-/ADU] and readout noise [e-] from a HDU header if present.
        Otherwise, use the current self.* values.
        """
        hdr = hdu.header
        if "gain" in self.keywords and self.keywords["gain"][0] in hdr:
            gain = hdr[self.keywords["gain"][0]]
        else:
            gain = self.gain
        if "ron" in self.keywords and self.keywords["ron"][0] in hdr:
            ron = hdr[self.keywords["ron"][0]]
        else:
            ron = self.ron
        if verbose:
            logging.info("gain : {0} e-/ADU".format(gain))
            logging.info("RON  : {0} e-".format(ron))
        return gain, ron

    def set_column_names(
        self, pixcol="pixel", flxcol="flux", errcol="err_flux", labcol="label"
    ):
        self.pixcol = pixcol
        self.flxcol = flxcol
        self.errcol = errcol
        self.labcol = labcol

    def extract(self, hdu, dark=None, method="simple") -> dict:
        """
        Extract the spectrum from an image in a FITS hdu into a list of Tables.
        If "dark" is given, this bias-subtracted data in HDU form is used to
        compute the additional noise.  The "simple" method uses Gaussian
        weighting along the path provided by the tracer.  The "integrated"
        method uses integrated Gaussian weighting along the path provided by
        the tracer.  The "optimal" method is similar to Horne (1986), as
        modified by Sandin et al. (2010).
        """
        gain, ron = self.get_gain_ron(hdu, verbose=True)  # JUST FOR logging

        if method not in ["simple", "integrated"]:
            raise NotImplementedException(
                "{0} extraction method not implemented!".format(method)
            )

        if method == "simple":
            return self._simple_extract(hdu, dark=dark, integrate=False)
        elif method == "integrate":
            return self._simple_extract(hdu, dark=dark, integrate=True)

    def _simple_extract(self, hdu, dark=None, integrate=False) -> list:
        """
        Simple weighted extraction.
        """
        gain, ron = self.get_gain_ron(hdu)  # e-/ADU,e-

        # GET DATA TO BE EXTRACTED (MUST BE BIAS-SUBTRACTED!)
        data = np.abs(hdu.data)
        ny, nx = data.shape
        if dark is not None:
            darkdata = np.abs(dark.data)
            if darkdata.shape != data.shape:
                raise ValueError("size of data does not match size of dark")
            noise = np.sqrt(ron**2 + gain * (data + darkdata)) / gain  # ADU
        else:
            noise = np.sqrt(ron**2 + gain * data) / gain  # ADU
        xdata = np.arange(nx)

        # FOR EACH TRACED SPECTRUM ...
        ns = self.tracer.number_of_traces()
        if ns <= 0:
            raise ValueError("no traced spectra!")
        spectra = []
        fibre = None
        next_fibre = self.tracer.get_fibre(0)
        for idx in range(0, ns):
            hdr = {}

            # ... GET NEIGHBORING FIBRES
            last_fibre = fibre
            fibre = next_fibre
            if idx <= ns:
                next_fibre = self.tracer.get_fibre(idx + 1)
            else:
                next_fibre = None

            # ... GET AMPLITUDES, SIGMAS, AND MODELS
            sig = fibre.sigma
            if sig is None:
                raise ValueError("sigma={1} for fibre #{2}".format(sig, idx))
            isigma = int(sig)
            pc, fc = fibre.get_trace_position_model()
            pa, fa = fibre.get_trace_amplitude_model()
            ycent = fc(xdata, *pc)  # FITTED CENTERS    OF SPECTRA
            amp = fa(xdata, *pa)  # FITTED AMPLITUDES OF SPECTRA

            if (
                last_fibre is None
                or last_fibre.ampl_coef is None
                or last_fibre.sigma is None
                or last_fibre.trace_coef is None
            ):
                last_sig = self.sigma
                last_pc, last_fc = pc, fc
                last_pa, last_fa = pa, fa
                last_ycent = ycent
                last_amp = amp
            else:
                last_sig = last_fibre.sigma
                last_pc, last_fc = last_fibre.get_trace_position_model()
                last_pa, last_fa = last_fibre.get_trace_amplitude_model()
                last_ycent = last_fc(xdata, *last_pc)
                last_amp = last_fa(xdata, *last_pa)

            if (
                next_fibre is None
                or next_fibre.ampl_coef is None
                or next_fibre.sigma is None
                or next_fibre.trace_coef is None
            ):
                next_sig = self.sigma
                next_pc, next_fc = pc, fc
                next_pa, next_fa = pa, fa
                next_ycent = ycent
                next_amp = amp
            else:
                next_sig = next_fibre.sigma
                next_pc, next_fc = next_fibre.get_trace_position_model()
                next_pa, next_fa = next_fibre.get_trace_amplitude_model()
                next_ycent = next_fc(xdata, *next_pc)
                next_amp = next_fa(xdata, *next_pa)

            # ... GET VALUE FOR EACH HORIZONTAL PIXEL
            fdata = np.zeros(nx)
            edata = np.zeros(nx)

            erfs = sig * np.sqrt(2.0)  # FUNNY NORMALIZATION OF erf()
            last_erfs = last_sig * np.sqrt(2.0)
            next_erfs = next_sig * np.sqrt(2.0)

            for i in range(nx):  # FOR ALL X-PIXELS
                x = float(i)
                yc = ycent[i]
                ac = amp[i]
                last_yc = last_ycent[i]
                last_ac = last_amp[i]
                next_yc = next_ycent[i]
                next_ac = next_amp[i]

                # GET REGION TO EXTRACT
                j = int(yc)
                j1 = j - isigma * 3 - 1  # 1 PIXEL EXTRA FOR FRACTIONAL PIXELS
                if j1 < 0:
                    j1 = 0
                j2 = j + isigma * 3 + 1
                if j2 >= ny:
                    j2 = ny - 1
                n = j2 - j1 + 1
                if j1 < 0 or j2 >= ny or n < 1:
                    fdata[i] = np.nan
                    edata[i] = np.nan

                else:
                    # GET EXTRACTED DATA PROFILE
                    yp = np.linspace(j1, j2, n)
                    pdata = data[j1 : j2 + 1, i]
                    pnoise = noise[j1 : j2 + 1, i]

                    # USE SIMPLE GAUSSIAN WEIGHTING
                    if not integrate:
                        prof = ac * np.exp(-((yc - yp) ** 2) / sig**2)
                        last_prof = last_ac * np.exp(
                            -((last_yc - yp) ** 2) / last_sig**2
                        )
                        next_prof = next_ac * np.exp(
                            -((next_yc - yp) ** 2) / next_sig**2
                        )

                    # OR INTEGRATED GAUSSIAN WEIGHTING 0.5*(1-erf(dx/sqrt(2))), dx=(y-ycenter)/sigma
                    else:
                        prof = (
                            ac
                            * 0.5
                            * (
                                erf((yp - yc + 0.5) / erfs)
                                - erf((yp - yc - 0.5) / erfs)
                            )
                        )
                        last_prof = (
                            last_ac
                            * 0.5
                            * (
                                erf((yp - last_yc + 0.5) / last_erfs)
                                - erf((yp - last_yc - 0.5) / last_erfs)
                            )
                        )
                        next_prof = (
                            next_ac
                            * 0.5
                            * (
                                erf((yp - next_yc + 0.5) / next_erfs)
                                - erf((yp - next_yc - 0.5) / next_erfs)
                            )
                        )
                    wgt = prof / (last_prof + prof + next_prof)
                    wgtsum = np.nansum(wgt)
                    flux = np.nansum(wgt * pdata) / wgtsum
                    f2err = np.nansum(wgt**2 * pnoise**2)
                    ferr = np.sqrt(f2err) / wgtsum
                    fdata[i] = flux
                    edata[i] = ferr

            # ... CREATE A NEW TABLE FRO THE EXTRACTED SPECTRUM
            t = Table()
            t[self.pixcol] = Column(xdata, unit="pix", description="pixel_position")
            t[self.flxcol] = Column(fdata, unit="adu", description="flux/pixel")
            t[self.errcol] = Column(edata, unit="adu", description="err_flux/pixel")
            t[self.idxcol] = Column(idx, unit="", description="rank")

            # ADD METADATA
            if not self.filter.copy_header(
                hdu.header, t.meta, bare=True, comments=False
            ):
                logging.warning("could not copy FITS header to table metadata")
            fibre.update_header(t.meta)

            # ADD TO LIST OF EXTRACTED SPECTRA
            spectra.append(t)
        return spectra


def main():
    import argparse, yaml
    from pyFU.utils import write_tables, parse_arguments, initialize_logging
    from pyFU.display import show_hdu

    # ---- READ CONFIGURATION AND COMMAND LINE PARAMETERS
    arguments = {
        "errcol": {
            "path": "extract:",
            "default": "err_flux",
            "flg": "-E",
            "type": str,
            "help": "name of flux error table column",
        },
        "vflip": {
            "path": "extract:",
            "default": False,
            "flg": "-V",
            "type": bool,
            "help": "flip input image vertically before extracting",
        },
        "flxcol": {
            "path": "extract:",
            "default": "flux",
            "flg": "-F",
            "type": str,
            "help": "name of flux table column",
        },
        "formats": {
            "path": None,
            "default": None,
            "flg": "-f",
            "type": str,
            "help": "YAML configuation file for FITS keyword formats",
        },
        "generic": {
            "path": None,
            "default": None,
            "flg": "-G",
            "type": str,
            "help": "YAML file for generic extraction configuration info",
        },
        "idxcol": {
            "path": "extract:",
            "default": "index",
            "flg": "-I",
            "type": str,
            "help": "name of index table column",
        },
        "infiles": {
            "path": "extract:",
            "default": None,
            "flg": "-i",
            "type": str,
            "help": "input FITS file names (default None)",
        },
        "keywords": {
            "path": None,
            "default": None,
            "flg": "-K",
            "type": str,
            "help": "YAML configuation file for FITS keywords",
        },
        "outfiles": {
            "path": "extract:",
            "default": None,
            "flg": "-o",
            "type": str,
            "help": "output FITS file names (default None)",
        },
        "pixcol": {
            "path": "extract:",
            "default": "pixel",
            "flg": "-P",
            "type": str,
            "help": "name of pixel table column",
        },
        "plot": {
            "path": None,
            "default": False,
            "flg": "-p",
            "type": bool,
            "help": "plot details",
        },
        "trace": {
            "path": "trace:save",
            "default": None,
            "flg": "-T",
            "type": str,
            "help": "path of (optional) YAML trace configuration file",
        },
        "yaml": {
            "path": None,
            "default": None,
            "flg": "-y",
            "type": str,
            "help": "path of (optional) generic YAML configuration file",
        },
    }
    args, cfg = parse_arguments(arguments)
    info = cfg["extract"]

    # ---- LOGGING
    initialize_logging(config=cfg)
    logging.info("*************************** extract ******************************")

    # ---- OUTPUT GENERIC CONFIGURATION FILE?
    if args.generic is not None:
        logging.info(
            "Appending generic extraction configuration info to" + args.generic
        )
        with open(args.generic, "a") as stream:
            yaml.dump({"extract": info}, stream)
        sys.exit(0)

    # ---- GET EXTERNAL KEYWORD DATA
    formats = None
    if args.formats is not None:
        with open(args.formats) as stream:
            formats = yaml.safe_load(stream)
    keywords = None
    if args.keywords is not None:
        with open(args.keywords) as stream:
            keywords = yaml.safe_load(stream)

    # ---- GET EXTERNAL TRACE CONFIGURATION
    if args.trace is not None:
        try:
            with open(args.trace) as stream:
                tracecfg = yaml.safe_load(stream)
        except:
            logging.error("cannot read trace YAML file " + args.trace + " !")
            sys.exit(1)
        merge_dictionaries(cfg, tracecfg)

    # ---- GET INPUT AND OUTPUT FILE NAMES
    infiles, outfiles = get_infiles_and_outfiles(args.infiles, args.outfiles, cfg=info)

    # ---- FOR ALL INPUT FILES
    for infile, outfile in zip(infiles, outfiles):
        logging.info(f"Extracting {outfile} from {infile}")

        # ---- READ IN DATA
        hdus = fits.open(infile)
        hdu = hdus[0]

        # ---- OPTIONAL VERTICAL FLIP
        if args.vflip:
            hdu.data = np.flipud(hdu.data)

        # ---- CREATE TRACER
        logging.info("Creating tracer...")
        if args.trace is not None:  # FROM TRACE YAML
            tracer = SpectrumTracer(
                hdu,
                config=cfg,
                formats=formats,
                keywords=keywords,
                external_traces=args.trace,
            )
        else:  # FROM FITS HEADER OF TRACED IMAGE OR CONFIG
            tracer = SpectrumTracer(hdu, config=cfg, formats=formats, keywords=keywords)

        # ---- PLOT HDU IMAGE AND HORIZONTAL TRACES
        if args.plot:
            logging.info("Plotting traces...")
            tracer.plot_traces(show_data=True, kappa=0.5)

        # ---- EXTRACT TRACED SPECTRA
        logging.info("Extracting spectra...")
        extractor = SpectrumExtractor(
            tracer, config=cfg, keywords=keywords, formats=formats
        )
        spectra = extractor.extract(hdu)
        if len(spectra) == 0:
            logging.error("No spectra extracted!")
            sys.exit(1)
        shift = np.mean(spectra[0][info["flxcol"]] * 0.5)

        # ---- PLOT RESULTS
        if args.plot:
            plt.figure()
            for i in range(len(spectra) - 1, -1, -1):
                s = spectra[i]
                x = s[info["pixcol"]].data
                f = s[info["flxcol"]].data + i * shift
                e = s[info["errcol"]].data
                lab = "#{0}".format(i + 1)
                fibre = tracer.get_fibre(i)
                lab = fibre.label
                plt.errorbar(x, f, yerr=e, fmt="-", label=lab)
            plt.xlabel("x [pix]")
            plt.ylabel("flux [ADU] + constant")
            plt.title(infile)
            ncol = 1
            if len(spectra) > 10:
                ncol = 2
            plt.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                ncol=ncol,
                fontsize="x-small",
            )
            plt.tight_layout()
            plt.show()

        # GET STRIPPED HEADER
        logging.info("Transfering header ...")
        hdr = {}
        extractor.filter.copy_header(hdu.header, hdr, bare=True, comments=False)

        # WRITE TO FITS BINARY TABLE FILE
        logging.info("Saving to FITS table file {0} ...".format(outfile))
        write_tables(
            spectra, outfile, header=hdr, overwrite=True, keywords=extractor.keywords
        )

        logging.info(
            "******************************************************************\n"
        )


if __name__ == "__main__":
    main()
