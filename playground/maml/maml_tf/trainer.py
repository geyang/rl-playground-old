import tensorflow as tf
from collections import defaultdict
from ml_logger import logger
from comet_ml import Experiment
from .e_maml_ge import E_MAML
from .ge_policies import MlpPolicy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from .config import RUN, G, DEBUG, Reporting
from .ge_utils import defaultlist, probe_var
import numpy as np
from . import ppo, vpg

comet_logger = Experiment(api_key="ajVBg1bSCmLJQ2aiCQu6Sp6aA", project_name='rl-playground/rl-maml')


class Trainer(object):
    def sample_from_env(self, env: SubprocVecEnv, policy: MlpPolicy, timestep_limit=None, render=False):
        """
        return: dimension is Size(timesteps, n_envs, feature_size)
        """
        # todo: use a default dict for these data collection. Much cleaner.
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []

        dones = [False] * env.num_envs
        if render:
            env.render()
        # while sum(dones) < env.num_envs:
        for _ in range(timestep_limit or G.batch_timesteps):
            # M.red("obs shape is: {}, value is: {}".format(self.obs.shape, self.obs))
            try:
                obs = self.obs
            except AttributeError:
                obs = self.obs = env.reset()
            actions, values, neglogpacs = policy.step(obs)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(dones)
            self.obs[:], rewards, dones, infos = env.step(actions)
            if render: env.render()
            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = policy.value(self.obs)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        last_gae_lam = 0
        n_rollouts = len(mb_obs)
        for t in reversed(range(n_rollouts)):
            if t == n_rollouts - 1:
                next_non_terminal = 1.0 - dones  # np.array(self.dones, dtype=float)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - mb_dones[t + 1]
                next_values = mb_values[t + 1]
            delta = mb_rewards[t] + G.gamma * next_values * next_non_terminal - mb_values[t]
            mb_advs[t] = last_gae_lam = delta + G.gamma * G.lam * next_non_terminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        # return dimension is Size(timesteps, n_envs, feature_size)
        return dict(obs=mb_obs, rewards=mb_rewards, returns=mb_returns, dones=mb_dones, actions=mb_actions,
                    values=mb_values, neglogpacs=mb_neglogpacs)

    def train(self, *, tasks, maml: E_MAML):
        max_grad_steps = max(G.n_grad_steps, *G.eval_grad_steps)
        for epoch_ind in range(G.n_epochs):
            # should_save = (epoch_ind % Reporting.save_interval == 0) if Reporting.save_interval else False

            frac = 1.0 - (epoch_ind - 1.0) / G.n_epochs
            alpha_lr = G.alpha * frac
            beta_lr = G.beta * frac
            clip_range = G.clip_range * frac

            # Compute updates for each task in the batch
            # 0. save value of variables
            # 1. sample
            # 2. gradient descent
            # 3. repeat step 1., 2. until all gradient steps are exhausted.
            batch_data = defaultdict(list)

            print('save checkpoint')
            maml.save_checkpoint()
            load_ops = [] if DEBUG.no_weight_reset else [maml.cache.load]

            graph_branch = maml.graphs[0]
            for task_ind in range(G.n_tasks):
                feed_dict = {}
                if task_ind == 0:
                    gradient_sum_op = maml.gradient_sum.set_op
                else:
                    gradient_sum_op = maml.gradient_sum.add_op

                if (task_ind == 0 and epoch_ind == 0) or not DEBUG.no_task_resample:
                    print('===> re-sample tasks')
                    env = tasks.sample()

                for k in range(max_grad_steps + 1):  # 0 - 10 <== last one being the maml policy.
                    # collect samples from the environment
                    if not G.single_sampling or (k == 0 or k == G.n_grad_steps):
                        # M.print('$!#$@#$ sample from environment')
                        p = self.sample_from_env(env, maml.runner.policy, render=False)

                    avg_r = np.mean(p['rewards'])
                    episode_r = avg_r * tasks.spec.max_episode_steps  # default horizon for HalfCheetah

                    if k in G.eval_grad_steps:
                        batch_data[f"grad_{k}_step_reward"].append(
                            avg_r if Reporting.report_mean else episode_r)

                    comet_logger.log_metric(f"task_{task_ind}_grad_{k}_reward", episode_r, step=epoch_ind)

                    if episode_r < G.term_reward_threshold:
                        # todo: make this based on batch instead of a single episode.
                        print(episode_r)
                        raise RuntimeError('AVERAGE REWARD TOO LOW. Terminating the experiment.')

                    _p = {k: v for k, v in p.items() if k != "ep_infos"}

                    if k < max_grad_steps:
                        # M.red('....... Optimize Model')
                        runner_feed_dict = \
                            path_to_feed_dict(inputs=maml.runner.inputs, paths=_p, lr=alpha_lr, clip_range=clip_range)
                        grads_val = maml.runner.optim.run_optimize(feed_dict=runner_feed_dict)

                    if k < G.n_grad_steps:
                        # print(f'step == {k}, assign path data to worker inputs')
                        feed_dict.update(
                            path_to_feed_dict(inputs=graph_branch.workers[k].inputs, paths=_p, lr=alpha_lr,
                                              clip_range=clip_range))
                    elif k == G.n_grad_steps:
                        # print(f'step == {k}, assign path data to meta inputs')
                        # we don't treat the meta_input the same way even though we could. This is more clear to read.
                        # note: feed in the learning rate only later.
                        feed_dict.update(
                            path_to_feed_dict(inputs=graph_branch.meta.inputs, paths=_p, clip_range=clip_range))

                        # load from checkpoint before computing the meta gradient\nrun gradient sum operation
                        if load_ops:
                            tf.get_default_session().run(load_ops)
                        tf.get_default_session().run(gradient_sum_op, feed_dict)
                        if load_ops:
                            tf.get_default_session().run(load_ops)

                    # if k == max_grad_steps:
                    #     # M.red('....... Optimize Model')
                    #     runner_feed_dict = \
                    #         path_to_feed_dict(inputs=maml.runner.inputs, paths=_p, lr=alpha_lr, clip_range=clip_range)
                    #     _ = maml.runner.optim.run_optimize(feed_dict=runner_feed_dict)

            # Now compute the meta gradients.
            # note: runner shares variables with the MAML graph. Reload from state_dict
            print('load, meta update, save')
            # note: if max_grad_step is the same as n_grad_steps then no need here.
            tf.get_default_session().run(maml.meta_update_op, {maml.beta: beta_lr})
            tf.get_default_session().run(maml.cache.save)
            # tf.get_default_session().run(load_ops + [maml.meta_optimize_op, maml.cache.save], {maml.beta: beta_lr})

            for key in batch_data.keys():
                reduced = np.array(batch_data[key]).mean()
                logger.log_keyvalue(epoch_ind, key, reduced)
                comet_logger.log_metric(key, reduced, step=epoch_ind)


def path_to_feed_dict(*, inputs, paths, lr=None, **rest):
    if isinstance(inputs, vpg.Inputs):
        return vpg.path_to_feed_dict(inputs=inputs, paths=paths, lr=lr)  # kl limit etc
    elif isinstance(inputs, ppo.Inputs):
        return ppo.path_to_feed_dict(inputs=inputs, paths=paths, lr=lr, **rest)  # kl limit etc
    else:
        raise NotImplementedError("Input type is not recognised")


# debug only
def eval_tensors(*, variable, feed_dict):
    import baselines.common.tf_util as U
    return tf.get_default_session().run(variable, feed_dict)
