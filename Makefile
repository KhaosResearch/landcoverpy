.PHONY: clean

install:
	@python -m pip install slccw --upgrade

build:
	@python -m build
	
release:
	@python -m twine upload --skip-existing -r khaos dist/slccw*

clean:
	@rm -rf build dist .eggs *.egg-info

format: clean
	@python -m isort --profile black src/
	
.DEFAULT_GOAL :=
all: clean build release install