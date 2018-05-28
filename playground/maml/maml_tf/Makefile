ROOT_PATH=/home/ge/machine_learning/exploration_in_meta_learning_reproduction_code
goal-cheetah:
	PYTHONPATH=$(ROOT_PATH)/packages:$(ROOT_PATH)/e_maml_ge python ./experiments/goal-cheetah.py --plotting-server "localhost"
dave:
	dave --config-file "./scripts/breakout.yml"
pull:
	aws s3 sync s3://ge_dqn /Users/ge/data/rcall/s3/ge_dqn
data:
	open /Users/ge/data/rcall/s3/ge_dqn
