#!/bin/sh
# pyfu_bin_links.sh
#
# Creates executables from the core pyFU scripts (may have to use sudo!).
#
# EDIT THESE PATHS FOR YOUR OS AND PYTHON VERSION
#
PYFUBIN=~/bin
#PYFUBIN=/opt/local/bin
#
PYFUDIR=./
#PYFUDIR=~/Library/python/pyFU/pyFU
#
rm -f ${PYFUBIN}/ifubb
rm -f ${PYFUBIN}/ifucal
rm -f ${PYFUBIN}/ifudis
rm -f ${PYFUBIN}/ifuext
rm -f ${PYFUBIN}/ifufak
rm -f ${PYFUBIN}/ifufla
rm -f ${PYFUBIN}/ifuima
rm -f ${PYFUBIN}/ifulam
rm -f ${PYFUBIN}/ifumgf
rm -f ${PYFUBIN}/ifureb
rm -f ${PYFUBIN}/ifusol
rm -f ${PYFUBIN}/ifutra
rm -f ${PYFUBIN}/ifuwav
rm -f ${PYFUBIN}/ifutool

ln -s ${PYFUDIR}/barebones.py ${PYFUBIN}/ifubb
ln -s ${PYFUDIR}/calib.py   ${PYFUBIN}/ifucal
ln -s ${PYFUDIR}/display.py ${PYFUBIN}/ifudis
ln -s ${PYFUDIR}/extract.py ${PYFUBIN}/ifuext
ln -s ${PYFUDIR}/fake.py    ${PYFUBIN}/ifufak
#ln -s ${PYFUDIR}/flat.py    ${PYFUBIN}/ifufla
ln -s ${PYFUDIR}/image.py   ${PYFUBIN}/ifuima
#ln -s ${PYFUDIR}/mgauss.py  ${PYFUBIN}/ifumgf
ln -s ${PYFUDIR}/lamp.py    ${PYFUBIN}/ifulam
ln -s ${PYFUDIR}/rebin.py   ${PYFUBIN}/ifureb
ln -s ${PYFUDIR}/solar.py   ${PYFUBIN}/ifusol
ln -s ${PYFUDIR}/trace.py   ${PYFUBIN}/ifutra
ln -s ${PYFUDIR}/wavcal.py  ${PYFUBIN}/ifuwav
ln -s ${PYFUDIR}/../pyfu_tool.py ${PYFUBIN}/ifutool

chmod +x ${PYFUBIN}/ifubb
chmod +x ${PYFUBIN}/ifucal
chmod +x ${PYFUBIN}/ifudis
chmod +x ${PYFUBIN}/ifuext
chmod +x ${PYFUBIN}/ifufak
#chmod +x ${PYFUBIN}/ifufla
chmod +x ${PYFUBIN}/ifuima
chmod +x ${PYFUBIN}/ifulam
#chmod +x ${PYFUBIN}/ifumgf
chmod +x ${PYFUBIN}/ifureb
chmod +x ${PYFUBIN}/ifusol
chmod +x ${PYFUBIN}/ifutra
chmod +x ${PYFUBIN}/ifuwav
chmod +x ${PYFUBIN}/ifutool
