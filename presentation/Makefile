all: presentation.pdf

presentation.pdf: main.tex
	latexmk -pdf -jobname=$(basename $@) $<
	latexmk -c -jobname=$(basename $@) $<
