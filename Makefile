.PHONY: clean

install:
	@python -m pip install bd_lc_mediterranean --upgrade

build:
	@python -m build
	
release:
	@python -m twine upload --skip-existing -r khaos dist/bd_lc_mediterranean*

clean:
	@rm -rf build dist .eggs *.egg-info

format: clean
	@python -m isort --profile black src/
	
.DEFAULT_GOAL :=
all: clean build release install