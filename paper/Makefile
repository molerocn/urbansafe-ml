.PHONY: build

build:
	@mkdir -p build
	@latexmk -pdf -outdir=build main.tex

run: build
	@xdg-open >/dev/null 2>&1 build/main.pdf
