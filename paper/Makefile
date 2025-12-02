.PHONY: build

build:
	@mkdir -p build
	@latexmk -pdf -outdir=build main.tex

clean:
	@rm -rf build/

run: build
	@xdg-open >/dev/null 2>&1 build/main.pdf
