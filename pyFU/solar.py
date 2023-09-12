#!/usr/bin/env python3

# pyfu/solar.py

import logging
import numpy as np
import sys
import yaml

from astropy.table import Table, Column
from astropy import units as u
from matplotlib import pyplot as plt

from pyFU.utils import (
    hist_rebin,
    read_spectrum,
    parse_arguments,
    convolve_rebin,
    initialize_logging,
)

# from pyFU.defaults import pyFU_logging_level, pyFU_logging_format
# logging.basicConfig (level=pyFU_logging_level, format=pyFU_logging_format)


def main():
    # ---- PARSE COMMAND LINE

    # ---- GET PARAMETERS FROM CONFIGURATION FILE OR COMMAND LINE (LATTER HAS HIGHER PRIORITY)
    README = """
	Python script to convolve a high-resolution Solar spectrum to a lower resolution.
	"""
    arguments = {
        "atlas": {
            "path": "solar:",
            "default": None,
            "flg": "-a",
            "type": str,
            "help": "input solar reference atlas (FITS table)",
        },
        "dispersion": {
            "path": "solar:",
            "default": None,
            "flg": "-d",
            "type": float,
            "help": "output dispersion in nm/pixel",
        },
        "fcol": {
            "path": "solar:",
            "default": "flux",
            "flg": "-f",
            "type": str,
            "help": "name of input flux table column)",
        },
        "flxcol": {
            "path": "solar:",
            "default": "flux",
            "flg": "-F",
            "type": str,
            "help": "name of output flux table column)",
        },
        "generic": {
            "path": None,
            "default": None,
            "flg": "-G",
            "type": str,
            "help": "YAML file for generic solar spectrum configuration info",
        },
        "limits": {
            "path": "solar:",
            "default": None,
            "dshow": "all",
            "flg": "-l",
            "type": str,
            "help": "wavelength limits w1,w2 [nm] of output spectrum",
        },
        "npixels": {
            "path": "solar:",
            "default": None,
            "flg": "-n",
            "type": int,
            "help": "number of output pixels",
        },
        "outfile": {
            "path": "solar:",
            "default": None,
            "flg": "-o",
            "type": str,
            "help": "output FITS table",
        },
        "plot": {
            "path": None,
            "default": False,
            "flg": "-p",
            "type": bool,
            "help": "plot result",
        },
        "resolution": {
            "path": "solar:",
            "default": None,
            "dshow": "<wave>/(3pix*dispersion)",
            "flg": "-R",
            "type": float,
            "help": "spectral resolution R=lambda/dlambda",
        },
        "wcol": {
            "path": "solar:",
            "default": "wavelength",
            "flg": "-w",
            "type": str,
            "help": "name of input wavelength table column)",
        },
        "wavcol": {
            "path": "solar:",
            "default": "wavelength",
            "flg": "-W",
            "type": str,
            "help": "name of output wavelength table column)",
        },
        "wresolution": {
            "path": "solar:",
            "default": None,
            "dshow": "(3pix*dispersion)",
            "flg": "-r",
            "type": float,
            "help": "wavelength resolution",
        },
        "yaml": {
            "path": None,
            "default": None,
            "flg": "-y",
            "type": str,
            "help": "global YAML configuration file for parameters",
        },
    }
    args, cfg = parse_arguments(arguments, readme=README)
    info = cfg["solar"]

    # ---- LOGGING
    initialize_logging(config=cfg)
    logging.info("*************************** solar ******************************")

    # ---- OUTPUT GENERIC CONFIGURATION FILE?
    if args.generic is not None:
        logging.info("Appending generic solar spectrum configuration to" + args.generic)
        with open(args.generic, "a") as stream:
            yaml.dump({"solar": info}, stream)
        sys.exit(0)

    # ---- GET INPUT SOLAR ATLAS SPECTRUM
    if "atlas" not in info or info["atlas"] is None:
        logging.critical("No input atlas given!")
        sys.exit(1)
    logging.info("Getting the reference atlas {0} ...".format(info["atlas"]))
    spectrum = read_spectrum(info["atlas"])
    colnames = spectrum.colnames
    if info["wcol"] not in colnames or info["fcol"] not in colnames:
        logging.error("{0} or {1} not column names!".format(info["wcol"], info["fcol"]))
        sys.exit(1)
    w = spectrum[info["wcol"]]
    f = spectrum[info["fcol"]]
    if str(w.unit).lower() == "angstrom":
        logging.info("Converting from Angstroms to nanometers...")
        w *= 0.1
    mdisp = np.mean(np.diff(w))
    mresol = np.mean(w) / mdisp
    logging.info(
        "Input number of pixels is {0}, mean dispersion is {1:.5f}".format(
            len(w), mdisp
        )
    )
    logging.info(
        "Mean input pixel resolution R_pix=lambda/dlambda is {0:.5f}".format(mresol)
    )

    # ---- GET OUTPUT WAVELENGTHS, NUMBER OF PIXELS, AND OUTPUT DISPERSION
    if info["limits"] is None:
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
        info["npixels"] = int((wave2 - wave1) / info["dispersion"] + 0.5) + 1

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
    ) and ("npixels" in info and info["npixels"] is not None and info["npixels"] > 0):
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

    # ---- GET OUTPUT WAVELENGTHS AND RESOLUTION
    wave = np.linspace(wave1, wave2, npixels)
    if "wresolution" not in info or info["wresolution"] is None:
        if ("resolution" not in info) or info["resolution"] is None:
            logging.critical("no output resolution given!")
            sys.exit(10)
        info["wresolution"] = wavg / info["resolution"]
    R = wavg / info["wresolution"]

    logging.info("Number of output pixels: {0}".format(npixels))
    logging.info("Output dispersion: {0:.2f}".format(disp))
    logging.info(
        "Output pixel resolution lambda/dlam: {0:.4f}".format(np.mean(wave) / disp)
    )
    logging.info(
        "Output spectral resolution: R={0:.2f}, dwave={1:.2f} nm".format(
            R, info["wresolution"]
        )
    )

    # ---- CREATE DATA WITH OUTPUT RESOLUTION
    logging.info(
        "degrading reference spectrum to resolution {0:.3f} nm ...".format(
            info["wresolution"]
        )
    )
    try:
        flux = convolve_rebin(w, f, dlambda=info["wresolution"], wave_out=wave)
    except KeyboardInterrupt:
        sys.exit(1)

    # ---- PLOT RESULTS
    if args.plot:
        logging.info("Plotting results ...")
        plt.plot(wave, flux, "-")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Flux")
        plt.xlim([np.min(wave), np.max(wave)])
        plt.ylim(bottom=0.0)
        plt.title("Solar spectrum at R={0:.1f}".format(R))
        plt.show()

    # ---- SAVE RESULTS
    if info["outfile"] is not None:
        logging.info("Saving results to {0} ...".format(info["outfile"]))
        tab = Table()
        tab[info["wavcol"]] = Column(wave, description="wavelength", unit="nm")
        tab[info["flxcol"]] = Column(flux, description="flux")
        tab.meta["CDELT1"] = disp
        tab.meta["RESOLUTN"] = R
        tab.write(info["outfile"], overwrite=True)


if __name__ == "__main__":
    main()
