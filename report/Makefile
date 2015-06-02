all: SeonwookPark_BIWI_SA.pdf

SeonwookPark_BIWI_SA.pdf: main.tex
	latexmk -pdf -jobname=$(basename $@) $<
	latexmk -c -jobname=$(basename $@) $<
