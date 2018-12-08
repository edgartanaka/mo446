filename=report

compile:
	@echo "Compiling..."
	latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" $(filename)
    
	@echo "Process completed successfully"
	@echo -n "Cleaning auxiliary files...\n"
	@rm -f *.fls *.fdb_latexmk *.out *.aux *.alg *.acr *.dvi *.gls *.log *.bbl *.blg *.ntn *.not *.lof *.lot *.toc *.loa *.lsg *.nlo *.nls *.ilg *.ind *.ist *.glg *.glo *.xdy *.acn *.idx *.loq *~
	# @rm -f $(filename).pdf
	@echo "Process completed successfully"