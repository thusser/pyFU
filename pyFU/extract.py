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

    def _simple_extract(self, hdu, dark=None, integrate=True) -> list:
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
        n_traces = self.tracer.number_of_traces()
        if n_traces <= 0:
            raise ValueError("no traced spectra!")
        spectra = []
        fibre = None
        next_fibre = self.tracer.get_fibre(0)
        for trace_id in range(0, n_traces):
            # ... GET NEIGHBORING FIBRES
            last_fibre = fibre
            fibre = next_fibre
            if trace_id <= n_traces:
                next_fibre = self.tracer.get_fibre(trace_id + 1)
            else:
                next_fibre = None

            # ... GET AMPLITUDES, SIGMAS, AND MODELS
            sig = fibre.sigma
            if sig is None:
                raise ValueError("sigma={0} for fibre #{1}".format(sig, trace_id))
            vertical_width = int(fibre.sigma)
            position_coefficients, position_function = fibre.get_trace_position_model()
            amplitude_coefficients, amplitude_function = fibre.get_trace_amplitude_model()
            vertical_centers = position_function(xdata, *position_coefficients)  # FITTED CENTERS    OF SPECTRA
            amplitudes = amplitude_function(xdata, *amplitude_coefficients)  # FITTED AMPLITUDES OF SPECTRA

            if (
                last_fibre is None
                or last_fibre.ampl_coef is None
                or last_fibre.sigma is None
                or last_fibre.trace_coef is None
            ):
                last_vertical_width = self.sigma
                last_vertical_centers = vertical_centers
                last_amplitudes = amplitudes
            else:
                last_vertical_width = last_fibre.sigma
                last_position_coefficients, last_position_function = last_fibre.get_trace_position_model()
                last_amplitude_coefficients, last_amplitude_function = last_fibre.get_trace_amplitude_model()
                last_vertical_centers = last_position_function(xdata, *last_position_coefficients)
                last_amplitudes = last_amplitude_function(xdata, *last_amplitude_coefficients)

            if (
                next_fibre is None
                or next_fibre.ampl_coef is None
                or next_fibre.sigma is None
                or next_fibre.trace_coef is None
            ):
                next_vertical_width = self.sigma
                next_vertical_centers = vertical_centers
                next_amplitudes = amplitudes
            else:
                next_vertical_width = next_fibre.sigma
                next_position_coefficients, next_position_function = next_fibre.get_trace_position_model()
                next_amplitude_coefficients, next_amplitude_function = next_fibre.get_trace_amplitude_model()
                next_vertical_centers = next_position_function(xdata, *next_position_coefficients)
                next_amplitudes = next_amplitude_function(xdata, *next_amplitude_coefficients)

            # ... GET VALUE FOR EACH HORIZONTAL PIXEL
            fluxes = np.zeros(nx)
            flux_errors = np.zeros(nx)

            error_function = vertical_width * np.sqrt(2.0)  # FUNNY NORMALIZATION OF erf()
            last_error_function = last_vertical_width * np.sqrt(2.0)
            next_error_function = next_vertical_width * np.sqrt(2.0)

            for x_pixel in range(nx):  # FOR ALL X-PIXELS
                vertical_center = int(vertical_centers[x_pixel])
                amplitude = amplitudes[x_pixel]
                last_vertical_center = last_vertical_centers[x_pixel]
                last_amplitude = last_amplitudes[x_pixel]
                next_vertical_center = next_vertical_centers[x_pixel]
                next_amplitude = next_amplitudes[x_pixel]

                # GET REGION TO EXTRACT
                vertical_lower_limit = vertical_center - vertical_width - 1  # 1 PIXEL EXTRA FOR FRACTIONAL PIXELS
                if vertical_lower_limit < 0:
                    vertical_lower_limit = 0
                vertical_upper_limit = vertical_center + vertical_width + 1
                if vertical_upper_limit >= ny:
                    vertical_upper_limit = ny - 1
                n_pixels_inside = vertical_upper_limit - vertical_lower_limit + 1
                if vertical_lower_limit < 0 or vertical_upper_limit >= ny or n_pixels_inside < 1:
                    fluxes[x_pixel] = np.nan
                    flux_errors[x_pixel] = np.nan

                else:
                    # GET EXTRACTED DATA PROFILE
                    y_pixels = np.linspace(vertical_lower_limit, vertical_upper_limit, n_pixels_inside)
                    data_inside = data[vertical_lower_limit: vertical_upper_limit, x_pixel]
                    noise_inside = noise[vertical_lower_limit: vertical_upper_limit, x_pixel]

                    # USE SIMPLE GAUSSIAN WEIGHTING
                    if not integrate:
                        profile = amplitude * np.exp(-((vertical_center - y_pixels) ** 2) / sig ** 2)
                        last_profile = last_amplitude * np.exp(
                            -((last_vertical_center - y_pixels) ** 2) / last_vertical_width ** 2
                        )
                        next_profile = next_amplitude * np.exp(
                            -((next_vertical_center - y_pixels) ** 2) / next_vertical_width ** 2
                        )

                    # OR INTEGRATED GAUSSIAN WEIGHTING 0.5*(1-erf(dx/sqrt(2))), dx=(y-ycenter)/sigma
                    else:
                        profile = (
                                amplitude
                                * 0.5
                                * (
                                erf((y_pixels - vertical_center + 0.5) / error_function)
                                - erf((y_pixels - vertical_center - 0.5) / error_function)
                            )
                        )
                        last_profile = (
                                last_amplitude
                                * 0.5
                                * (
                                erf((y_pixels - last_vertical_center + 0.5) / last_error_function)
                                - erf((y_pixels - last_vertical_center - 0.5) / last_error_function)
                            )
                        )
                        next_profile = (
                                next_amplitude
                                * 0.5
                                * (
                                erf((y_pixels - next_vertical_center + 0.5) / next_error_function)
                                - erf((y_pixels - next_vertical_center - 0.5) / next_error_function)
                            )
                        )
                    weights = profile / (last_profile + profile + next_profile)
                    sum_of_weights = np.nansum(weights)
                    #flux = np.nansum(weights * pdata) / sum_of_weights   # TODO: think about correct weighting (current one leads to too small values)
                    #print(f'Trace from {vertical_upper_limit} to {j2}.')
                    flux = np.nansum(data_inside)
                    flux_err2 = np.nansum(weights ** 2 * noise_inside ** 2)
                    flux_error = np.sqrt(flux_err2) / sum_of_weights
                    fluxes[x_pixel] = flux
                    flux_errors[x_pixel] = flux_error

            # ... CREATE A NEW TABLE FRO THE EXTRACTED SPECTRUM
            t = Table()
            t[self.pixcol] = Column(xdata, unit="pix", description="pixel_position")
            t[self.flxcol] = Column(fluxes, unit="adu", description="flux/pixel")
            t[self.errcol] = Column(flux_errors, unit="adu", description="err_flux/pixel")
            t[self.idxcol] = Column(trace_id, unit="", description="rank")

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
        "packed_fits": {
            "path": None,
            "default": True,
            "flg": "-pck",
            "type": bool,
            "help": "True if input end with fits.fz, which is the case for pyobs-archive files",
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
            "path": "extract:",
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
        if info['packed_fits']:
            hdu = hdus['sci']
        else:
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
                f = s[info["flxcol"]].data# + i * shift
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
