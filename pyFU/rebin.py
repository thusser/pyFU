#!/usr/bin/env python3

# pyfu/rebin.py

import logging
import numpy as np
import sys
import yaml

from astropy.table import Table, Column
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from pyFU.display import show_with_menu
from pyFU.utils import (
    hist_rebin,
    read_tables,
    convolve_rebin_table,
    parse_arguments,
    write_tables,
    read_spectrum,
    get_infiles_and_outfiles,
    initialize_logging,
)

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)


def main():
    # ---- GET PARAMETERS FROM CONFIGURATION FILE OR COMMAND LINE (LATTER HAS HIGHER PRIORITY)
    README = """
	Python script to convolve them with a Gaussian profile to change their spectral resolution and/or to rebin them to a particular linear dispersion relation. 
	"""
    arguments = {
        "clobber": {
            "path": None,
            "default": False,
            "flg": "-c",
            "type": bool,
            "help": "permit overwriting of tables",
        },
        "dispersion": {
            "path": "rebin:",
            "default": None,
            "flg": "-d",
            "type": float,
            "help": "output dispersion in nm/pixel",
        },
        "errcol": {
            "path": "rebin:",
            "default": "err_flux",
            "flg": "-E",
            "type": str,
            "help": "name of flux error column)",
        },
        "flxcol": {
            "path": "rebin:",
            "default": "flux",
            "flg": "-F",
            "type": str,
            "help": "name of flux column)",
        },
        "generic": {
            "path": None,
            "default": None,
            "flg": "-G",
            "type": str,
            "help": "YAML file for generic rebin image configuration info",
        },
        "infiles": {
            "path": "rebin:",
            "default": None,
            "flg": "-i",
            "type": str,
            "help": "input FITS binary table(s) containing spectra",
        },
        "limits": {
            "path": "rebin:",
            "default": None,
            "flg": "-l",
            "type": str,
            "help": "wavelength limits w1,w2 [nm] of output spectrum",
        },
        "npixels": {
            "path": "rebin:",
            "default": None,
            "flg": "-n",
            "type": int,
            "help": "number of output pixels",
        },
        "outfiles": {
            "path": "rebin:",
            "default": None,
            "flg": "-o",
            "type": str,
            "help": "output FITS table(s)",
        },
        "plot": {
            "path": None,
            "default": False,
            "flg": "-p",
            "type": bool,
            "help": "plot result",
        },
        "pixcol": {
            "path": "rebin:",
            "default": "pixel",
            "flg": "-P",
            "type": str,
            "help": "name of pixel column)",
        },
        "title": {
            "path": None,
            "default": "rebin",
            "flg": "-T",
            "type": str,
            "help": "logging title)",
        },
        "reference": {
            "path": "rebin:",
            "default": None,
            "flg": "-R",
            "type": str,
            "help": "pathname of reference spectrum file",
        },
        "wavcol": {
            "path": "rebin:",
            "default": "wavelength",
            "flg": "-w",
            "type": str,
            "help": "name of wavelength column",
        },
        "wcol": {
            "path": "rebin:",
            "default": "wavelength",
            "flg": "-W",
            "type": str,
            "help": "name of wavelength column in reference",
        },
        "wresolution": {
            "path": "rebin:",
            "default": None,
            "flg": "-r",
            "type": float,
            "help": "wavelength resolution to apply",
        },
        "yaml": {
            "path": "rebin:",
            "default": None,
            "flg": "-y",
            "type": str,
            "help": "global YAML configuration file for parameters",
        },
    }
    args, cfg = parse_arguments(arguments, readme=README)
    info = cfg["rebin"]

    # ---- LOGGING
    initialize_logging(config=cfg)
    logging.info(
        f"*************************** {args.title} ******************************"
    )

    # ---- OUTPUT GENERIC CONFIGURATION FILE?
    if args.generic is not None:
        logging.info("Appending generic image configuration info to" + args.generic)
        with open(args.generic, "a") as stream:
            yaml.dump({"rebin": info}, stream)
        sys.exit(0)

    # ---- GET INPUT AND OUTPUT FILE NAMES
    infiles, outfiles = get_infiles_and_outfiles(args.infiles, args.outfiles, cfg=info)

    # ---- GET OUTPUT WAVELENGTHS VIA REFERENCE SPECTRUM OR NUMBER OF PIXELS, RANGE, AND/OR OUTPUT DISPERSION
    if "reference" in info and info["reference"] is not None:
        logging.info(f"using wavelengths of reference spectrum {info[reference]}")
        tab = read_spectrum(info["reference"])
        if (tab is None) or ("wcol" not in info) or (info["wcol"] not in tab.colnames):
            logging.error("unable to read wavelengths from reference spectrum")
            sys.exit(1)
        wave = tab[info["wcol"]]
        if str(wave.unit).lower() == "angstrom":
            logging.info("Converting from Angstroms to nanometers...")
            wave *= 0.1

    # ---- FOR ALL INPUT TABLES
    for infile, outfile in zip(infiles, outfiles):
        logging.info(f"Reading the table file {infile} ...")
        spectra, hdr = read_tables(infile)
        if len(spectra) == 0:
            logging.error(f"no spectra in {infile}")

        else:
            # GET WAVELENGTH LIMITS
            if ("limits" not in info) or (info["limits"] is None):
                logging.info(
                    "using wavelength bounds of #1/{0} spectrum".format(len(spectra))
                )
                spectrum = spectra[0]
                if "wavcol" not in info or info["wavcol"] not in spectrum.colnames:
                    logging.error("cannot access wavelengths in spectra")
                    sys.exit(1)
                w = spectrum[info["wavcol"]]
                wave1, wave2 = [w[0], w[-1]]
            else:
                if isinstance(info["limits"], str):
                    waves = info["limits"].split(",")
                    wave1, wave2 = float(waves[0]), float(waves[1])
                else:
                    wave1, wave2 = info["limits"]

            # ... MUST HAVE ENOUGH INFO
            if "npixels" not in info and "dispersion" not in info:
                logging.critical("neither dispersion nor number of pixels given!")
                print(info)
                sys.exit(1)
            # ... GET NUMBER OF PIXELS, FROM DISPERSION AND LIMITS
            elif (
                "npixels" not in info or info["npixels"] is None or info["npixels"] <= 0
            ) and ("dispersion" in info and info["dispersion"] is not None):
                info["npixels"] = (wave2 - wave1) / info["dispersion"] + 1

            # ... GET DISPERSION FROM NUMBER OF PIXELS AND LIMITS
            elif (
                "dispersion" not in info
                or info["dispersion"] is None
                or info["dispersion"] <= 0.0
            ) and ("npixels" in info and info["npixels"] is not None):
                info["dispersion"] = (wave2 - wave1) / (info["npixels"] - 1)

            # ... HAVE DISPERSION AND NPIXELS, SO USE STARTING WAVELENGTH
            elif (
                "dispersion" in info
                and info["dispersion"] is not None
                and info["dispersion"] > 0.0
            ) and (
                "npixels" in info
                and info["npixels"] is not None
                and info["npixels"] > 0
            ):
                wave2 = wave1 + (info["npixels"] - 1) * info["dispersion"]
                logging.warning(
                    "using given npixels, dispersion and resulting ending wavelength {0:.2f}!".format(
                        wave2
                    )
                )

            # ... OR WE HAVE A PROBLEM!
            else:
                logging.critical("unable to define dispersion or number of pixels!")
                print(info)
                sys.exit(1)

            npixels = info["npixels"]
            disp = info["dispersion"]
            wavg = 0.5 * (wave1 + wave2)
            wave = np.linspace(wave1, wave2, npixels)

        if "wresolution" in info and info["wresolution"] is not None:
            logging.info(
                "degrading input spectra to resolution {0:.3f} nm ...".format(
                    info["wresolution"]
                )
            )
        else:
            info["wresolution"] = None

        # ---- FOR EACH SPECTRUM...

        results = []
        for i in range(len(spectra)):
            spectrum = spectra[i]
            tab = convolve_rebin_table(
                spectrum,
                pixcol=info["pixcol"],
                wavcol=info["wavcol"],
                flxcol=info["flxcol"],
                errcol=info["errcol"],
                dlambda=info["wresolution"],
                wave_out=wave,
            )
            results.append(tab)

            # ---- PLOT RESULTS
            if args.plot:
                flux = tab[info["flxcol"]]
                fig = plt.figure()
                plt.tight_layout()
                if ("errcol" not in info) or (info["errcol"] is None):
                    plt.plot(
                        spectrum[info["wavcol"]],
                        spectrum[info["flxcol"]],
                        "-",
                        label="original",
                        drawstyle="steps-mid",
                    )
                    plt.plot(
                        wave, flux, ".", label="rebinned", markersize=4, color="black"
                    )
                else:
                    err = tab[info["errcol"]]
                    plt.plot(
                        spectrum[info["wavcol"]],
                        spectrum[info["flxcol"]],
                        "-",
                        label="original",
                        drawstyle="steps-mid",
                    )
                    plt.errorbar(
                        wave,
                        flux,
                        yerr=err,
                        fmt=".",
                        label="rebinned",
                        markersize=4,
                        color="black",
                    )
                plt.xlabel("Wavelength [nm]")
                plt.ylabel("Flux")
                plt.xlim([np.min(wave), np.max(wave)])
                plt.ylim(bottom=0.0)
                plt.title(f"Rebinned {infile} - #{i}")
                plt.legend()
                reslt = show_with_menu(fig, ["no more plots", "ABORT"])
                if reslt == "ABORT":
                    sys.exit(1)
                elif reslt == "no more plots":
                    args.plot = False

        # ---- SAVE RESULTS
        if outfile is not None:
            logging.info(f"Saving results to {outfile} ...")
            write_tables(results, pathname=outfile, overwrite=args.clobber)

        logging.info(
            "***********************************************************************\n"
        )


if __name__ == "__main__":
    main()
