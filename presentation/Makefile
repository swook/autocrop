all: presentation.pdf

presentation.pdf: main.tex
	latexmk -pdf -jobname=$(basename $@) $<
	latexmk -c -jobname=$(basename $@) $<
	mv $@ tmp.pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$@ tmp.pdf
	rm tmp.pdf
