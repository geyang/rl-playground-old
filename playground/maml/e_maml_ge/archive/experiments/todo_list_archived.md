# Experiments

There isn't a lot of time, I need to be much more efficient in executing experiments.

Right now the problem is that both the `maml.runner` and the `maml.meta_runner` are not working well on `HalfCheetan-v1`. I think these need to get done before `maml` or `e_maml` could work. On the other hand, if both of these work, then `maml` should work as intended. The fact that these are not working indicates that there is still something off in my implementation.

**Update**: The updated parallel (during back-prop) maml worked very well. Good results showed during parameter search.

## Todo

### Done
- [x] Make sure tensorflow `tf.gradient` is behaving as expected
- [x] check the weights are identical after the first step and the second
- check if maml.runner gives the correct parameter.
    - [x] should work with 1 PPO SGD + 1 PPO Adam
- [x] get `HalfCheetah` to work with meta runner
    - [x] with SGD (1.0) and Adam (0.003), both with 1 optimization episode.
- [x] get `GoalCheetah` to work with meta runner

1. [x] get PPO to run as well as reference implementation
   1. [x] run `rl_algs.PPO` as well as I can
   2. [x] get `maml.PPO` to work as well as `rl_algs.PPO`
   3. [x] investigate `Trainner`, see if env reset affects performance.
   4. use `maml.PPO` inside `rl_algs`
2. get `maml.runner_PPO` to run as well, inside the `e_maml` code base.
3. get `maml.meta_PPO` to run as well, inside the `e_maml` code base.

Then second stage:

4. get `maml` to run with:
   1. 1 task, 0 gradient steps on `HalfCheetah-v1`
   2. 1 task, 1 gradient step on `HalfCheetah-v1`
   3. 10 tasks, 1 gradient step on `HalfCheetah-v1`
5. Now with `HalfCheetahVoalVel-v0`
   1. 1 task, 0 gradient steps, no task reset
   2. 1 task, 1 gradient steps, no task reset
   3. 10 tasks, 1 gradient step, no task reset
   4. 10 tasks, 1 gradient step, task resampling.

