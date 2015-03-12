all: SeonwookPark_CGL_SA.pdf

SeonwookPark_CGL_SA.pdf: report.tex chapters/*
	latexmk -pdf -jobname=$(basename $@) $<
	latexmk -c -jobname=$(basename $@)
