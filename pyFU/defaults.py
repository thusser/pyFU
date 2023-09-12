# pyFU/defaults.py

# ---- LOGGING

import logging

pyFU_logging_level = logging.INFO
pyFU_logging_format = "%(asctime)s * %(levelname)s * %(message)s"
pyFU_logging_file = None

# ---- FORMATS

pyFU_default_formats = {
    "ampl_format": "TR-A{0:03d}{1:1x}",
    "trace_format": "TR-T{0:03d}{1:1d}",
    "sigma_format": "IF-SG{0:03d}",
    "inner_diameter_format": "IF-DI{0:03d}",
    "outer_diameter_format": "IF-DO{0:03d}",
    "index_format": "IF-ID{0:03d}",
    "label_format": "IF-LA{0:03d}",
    "spacing_format": "IF-SP{0:03d}",
    "xpos_format": "IF-XP{0:03d}",
    "ypos_format": "IF-YP{0:03d}",
}

# ---- KEYWORDS

pyFU_default_keywords = {
    "ampl_coef": (None, "ampl. coefficient a(index,hex_ord)"),
    "ampl_format": ("TR-AFRMT", "format of amplitude coef(trace,hex_order)"),
    "ampl_model": ("TR-AMODL", "model for trace amplitude"),
    "ampl_order": ("TR-AORDR", "order/n_parameters of amplitude fits"),
    "bias": ("CCD-BIAS", "CCD bias in e-/ADU"),
    "bkg_factor": ("TR-FBKGD", "factor used to exclude low-level max."),
    "biassec": ("BIASSEC", "raw region containing the bias overscan"),
    "dy_max": ("TR-DYMAX", "max. allowed vert. dev. of local max."),
    "dy_min": ("TR-DYMIN", "min. allowed vert. dev. of local max."),
    "exptime": ("EXPTIME", "exposure time [s]"),
    "gain": ("CCD-GAIN", "gain of CCD [e-/ADU]"),
    "index": ("IF-INDEX", "index of a single traced spectrum"),
    "index_format": ("IF-IDFMT", "format for reading fibre indices"),
    "inner_diameter": ("IF-DINNR", "inner diameter of a single fibre"),
    "inner_diameter_format": ("IF-DIFMT", "format for reading inner diameters"),
    "label": ("IF-LABEL", "label of a single spectrum"),
    "label_format": ("IF-LAFMT", "format for reading fibre labels"),
    "mid_slice": ("TR-SLICE", "starting vert. slice to find local max."),
    "number_slices": ("TR-NSLIC", "number of vert. slices"),
    "number_traces": ("TR-NUMBR", "(desired) number of horiz. traces"),
    "outer_diameter": ("IF-DOUTR", "outer diameter of a single fibre"),
    "outer_diameter_format": ("IF-DOFMT", "format for reading outer diameters"),
    "pixel1": ("IF-LOPIX", "lower integration range [pix]"),
    "pixel2": ("IF-HIPIX", "upper integration range [pix]"),
    "ron": ("CCD-RON", "readout noise of CCD [e-]"),
    "sigma": ("TR-SIGMA", "median vertical width of traces"),
    "sigma_factor": ("TR-SFACT", "sigma factor for fat fibres"),
    "sigma_format": ("IF-SGFMT", "format for reading fibre sigmas"),
    "sigma_kappa": ("TR-SKAPA", "kappa clipping for finding fat fibres"),
    "sigma_fit0": ("TR-SIGC0", "c0, sigma=c0+c1*x+c2*x^2+c3*x^3"),
    "sigma_fit1": ("TR-SIGC1", "c1, sigma=c0+c1*x+c2*x^2+c3*x^3"),
    "sigma_fit2": ("TR-SIGC1", "c2, sigma=c0+c1*x+c2*x^2+c3*x^3"),
    "sigma_fit3": ("TR-SIGC1", "c3, sigma=c0+c1*x+c2*x^2+c3*x^3"),
    "sigma_order": ("TR-SORDR", "order of sigma(x) polynomial"),
    "spacing": ("IF-SPACE", "median vertical spacing of spectra"),
    "spacing_format": ("IF-SPFMT", "format for reading fibre spacings"),
    "trace_bias": ("TR-BKGND", "assumed trace background"),
    "trace_coef": (None, "trace coefficient c(index,ord)"),
    "trace_format": ("TR-TCFMT", "format of trace coef(trace,order)"),
    "trace_index": ("TR-INDEX", "index of traced spectrum"),
    "trace_order": ("TR-TORDR", "polynomial order of trace position fit"),
    "trimsec": ("TRIMSEC", "raw region containing actual data"),
    "wave1": ("IF-LOWAV", "lower integration range [nm]"),
    "wave2": ("IF-HIWAV", "upper integration range [nm]"),
    "window_centroid": ("TR-WCNTD", "width of vert. centroid window"),
    "window_max": ("TR-WMAXI", "width of vert. window used to find max."),
    "window_profile": ("TR-WPROF", "width of vert. profile window"),
    "x_max": ("TR-XMAXI", "maximum traced x [pix]"),
    "x_min": ("TR-XMINI", "minimum traced x [pix]"),
    "xpos": ("IF-XPOSI", "focal plane x-pos. of a single fibre"),
    "xpos_format": ("IF-XPFMT", "format for reading focal plane pos."),
    "ypos": ("IF-YPOSI", "focal plane y-pos. of a single fibre"),
    "ypos_format": ("IF-YPFMT", "format for reading focal plane pos."),
}
