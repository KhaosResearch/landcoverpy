The package greensenti is available in the Khaos (private) PyPi repo. 

For build the package:
`python3 -m build && python3 -m twine upload --skip-existing -r khaos dist/etc_workflow* && pip install etc_workflow --upgrade`