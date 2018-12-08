#!/usr/bin/env bash

# builds report.pdf
#docker run --rm -it -v "$(pwd)":/home adnrv/texlive latexmk -pdf -pdflatex="pdflatex --shell-escape %O %S" report
docker run --rm -it -v "$(pwd)":/home adnrv/texlive report/make