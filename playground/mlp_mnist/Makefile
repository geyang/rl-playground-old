#SHELL=/bin/bash
SHELL=/usr/local/bin/bash
all:
	source activate playground &&\
	which python &&\
	python -m playground.mlp_mnist.__init__ --model=mlp &&\
	python -m playground.mlp_mnist.__init__ --model=lenet
