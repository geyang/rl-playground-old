# shell option to use extended glob from from https://stackoverflow.com/a/6922447/1560241
SHELL:=/bin/bash -O extglob

author=$(Ge Yang)
author_email=$(yangge1987@gmail.com)

.PHONY: build
build:
	docker-compose up --build

# notes on python packaging: http://python-packaging.readthedocs.io/en/latest/minimal.html
conda-env:
	conda env create -f ./playground.yml
default: ;
wheel:
	rm -rf dist
	python setup.py bdist_wheel
dev:
	make wheel
	pip install --ignore-installed dist/playground*.whl
convert-rst:
	pandoc -s README.md -o README --to=rst
	sed -i '' 's/code/code-block/g' README
	sed -i '' 's/\.\. code-block:: log/.. code-block:: text/g' README
	sed -i '' 's/\.\//https\:\/\/github\.com\/episodeyang\/playground\/blob\/master\//g' README
	perl -p -i -e 's/\.(jpg|png|gif)/.$$1?raw=true/' README
	rst-lint README
resize: # from https://stackoverflow.com/a/28221795/1560241
	echo ./figures/!(*resized).jpg
	convert ./figures/!(*resized).jpg -resize 888x1000 -set filename:f '%t' ./figures/'%[filename:f]_resized.jpg'
update-doc: convert-rst
	python setup.py sdist upload
publish: convert-rst
	make test
	make wheel
	twine upload dist/*
test:
	cd playground && \
	pwd && \
	python -m pytest --capture=no
