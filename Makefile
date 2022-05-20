install:
	@python -m pip install install etc_workflow --upgrade

build:
	@python -m build
	
release:
	@python -m twine upload --skip-existing -r khaos dist/etc_workflow*

clean:
	@rm -rf build dist .eggs *.egg-info
	
.DEFAULT_GOAL :=
all: clean build release install