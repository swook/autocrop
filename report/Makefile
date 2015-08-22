all: SeonwookPark_BIWI_SA.pdf

dev: dev.pdf

dev.pdf: main.tex *.tex
	latexmk -pdf -jobname=dev $<
	latexmk -c -jobname=dev $<

SeonwookPark_BIWI_SA.pdf: main.tex *.tex
	latexmk -pdf -jobname=$(basename $@) $<
	latexmk -c -jobname=$(basename $@) $<
	mv $@ tmp.pdf
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$@ tmp.pdf
	rm tmp.pdf

clean:
	rm -f *.pdf
