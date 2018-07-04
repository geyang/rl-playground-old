from collections import defaultdict
from ml_logger import logger
from comet_ml import Experiment
from .e_maml_ge import E_MAML
from .ge_policies import MlpPolicy
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from .config import RUN, G, DEBUG, Reporting
from .ge_utils import defaultlist
import numpy as np
from . import ppo, vpg


comet_logger = Experiment(api_key="ajVBg1bSCmLJQ2aiCQu6Sp6aA", project_name='rl-playground/rl-maml')


class Trainer(object):
    def sample_from_env(self, env: SubprocVecEnv, policy: MlpPolicy, timestep_limit=None, render=False):
        # todo: use a default dict for these data collection. Much cleaner.
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []

        dones = [False] * env.num_envs
        if render: env.render()
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

        def sf01(arr):
            """swap and then flatten axes 0 and 1"""
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        mb_obs, mb_rewards, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs = \
            map(sf01, (mb_obs, mb_rewards, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs))
        return dict(obs=mb_obs, rewards=mb_rewards, returns=mb_returns, dones=mb_dones, actions=mb_actions,
                    values=mb_values, neglogpacs=mb_neglogpacs)

    def train(self, *, tasks, maml: E_MAML, plot_fn=None, test_tasks=None):
        max_grad_steps = max(G.n_grad_steps, *G.eval_grad_steps)
        for epoch_ind in range(G.n_epochs):

            is_the_end = (epoch_ind == G.n_epochs)
            should_plot = (epoch_ind % Reporting.plot_interval == 0) if Reporting.plot_interval else False
            should_save = (epoch_ind % Reporting.save_interval == 0) if Reporting.save_interval else False
            should_test = (epoch_ind % G.eval_test_interval == 0) if G.eval_test_interval else False

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

            if DEBUG.debug_params:
                debug_tensor_key = 'runner_network/MlpPolicy/pi/b:0'
                runner_state_dict = {}
                meta_state_dict = {}
                runner_grads = defaultlist(dict)
                meta_grads = defaultlist(dict)

            all_grads = []
            if not DEBUG.no_weight_reset:
                # M.white('<--- save weights')
                maml.save_checkpoint()

            feed_dict = {}
            for task_ind, meta_branch in enumerate(maml.task_graphs):
                if not DEBUG.no_task_resample or (task_ind == 0 and epoch_ind == 0):
                    # M.white('===> re-sample tasks', end='')
                    env = tasks.sample()
                if task_ind != 0 and not DEBUG.no_weight_reset:
                    # M.white('---> resetting weights for worker sampling')
                    maml.load_checkpoint()
                else:
                    # M.white('---> Do NOT reset for first worker')
                    pass

                worker_paths = defaultlist(None)  # get paths for the first update.
                for k in range(max_grad_steps + 1):  # 0 - 10 <== last one being the maml policy.
                    # debug code
                    if DEBUG.debug_params:
                        runner_state_dict[k] = maml.runner.policy.state_dict
                        print("k =", k, debug_tensor_key, ": ", end='')
                        print(runner_state_dict[k][debug_tensor_key])

                    # collect samples from the environment
                    if G.single_sampling:
                        if k == 0 or k == G.n_grad_steps:
                            # M.print('$!#$@#$ sample from environment')
                            worker_paths[k] = p = self.sample_from_env(env, maml.runner.policy, render=False)
                        else:
                            # M.print('^^^^^^^ copy previous sample')
                            worker_paths[k] = p
                    else:
                        # M.print('$!#$@#$ sample from environment')
                        worker_paths[k] = p = self.sample_from_env(env, maml.runner.policy, render=False)

                    avg_r = np.mean(p['rewards'])
                    episode_r = avg_r * tasks.spec.max_episode_steps  # default horizon for HalfCheetah

                    if k in G.eval_grad_steps:
                        batch_data['grad_{}_step_reward'.format(k)].append(
                            avg_r if Reporting.report_mean else episode_r)

                    if episode_r < G.term_reward_threshold:
                        # todo: make this based on batch instead of a single episode.
                        print(episode_r)
                        raise RuntimeError('AVERAGE REWARD TOO LOW. Terminating the experiment.')

                    _p = {k: v for k, v in p.items() if k != "ep_infos"}

                    # Here we gradient descent on the same data only once. In the future, we could explore case
                    # involving more updates.
                    if k < max_grad_steps:
                        # M.red('....... Optimize Model')
                        runner_feed_dict = \
                            path_to_feed_dict(inputs=maml.runner.inputs, paths=_p, lr=alpha_lr, clip_range=clip_range)

                        if not DEBUG.debug_params:
                            maml.runner.optim.run_optimize(feed_dict=runner_feed_dict)

                        if DEBUG.debug_params:
                            _grads, *_ = maml.runner.model.run_grads(feed_dict=runner_feed_dict)
                            runner_grads[k] = {t.name: g for t, g in zip(maml.runner.policy.trainables, _grads)}
                            print('runner_grads:', runner_grads[k][debug_tensor_key])
                            if DEBUG.debug_apply_gradient:
                                maml.runner.optim.run_apply_grads(grads=_grads, lr=alpha_lr)
                            else:
                                maml.runner.optim.run_optimize(feed_dict=runner_feed_dict)

                    if k < G.n_grad_steps:
                        feed_dict.update(
                            path_to_feed_dict(inputs=meta_branch.workers[k].inputs, paths=_p, lr=alpha_lr,
                                              clip_range=clip_range))
                    elif k == G.n_grad_steps:
                        # we don't treat the meta_input the same way even though we could. This is more clear to read.
                        # note: feed in the learning rate only later.
                        feed_dict.update(
                            path_to_feed_dict(inputs=meta_branch.meta.inputs, paths=_p, clip_range=clip_range))

            # Now compute the gradients.
            # note: runner shares variables with the MAML graph. Reload from state_dict
            # note: should use variable placeholders for these inputs.
            if not DEBUG.no_weight_reset:
                from moleskin import moleskin as M
                M.green('---> resetting weights for meta gradient')
                maml.load_checkpoint()

            feed_dict[maml.beta] = beta_lr
            maml.optim.run_optimize(feed_dict=feed_dict)

            for key in batch_data.keys():
                reduced = np.array(batch_data[key]).mean()
                logger.log_keyvalue(epoch_ind, key, reduced)
                comet_logger.log_metric(key, reduced, step=epoch_ind)

            if should_test and test_tasks is not None:
                maml.save_checkpoint()
                print(test_tasks.spec)
                test_envs = test_tasks.envs
                test_envs.reset()
                p = self.sample_from_env(test_envs, maml.runner.policy, timestep_limit=test_tasks.spec.timestep_limit)
                logger.log(epoch_ind, pre_update_rewards=np.mean(p['rewards']))
                comet_logger.log_metric(pre_update_rewards, np.mean(p['rewards']), step=epoch_ind)
                p = self.sample_from_env(test_envs, maml.runner.policy)
                runner_feed_dict = \
                    path_to_feed_dict(inputs=maml.runner.inputs, paths=p, lr=alpha_lr, clip_range=clip_range)
                maml.runner.model.run_optimize(feed_dict=runner_feed_dict)
                p = self.sample_from_env(test_envs, maml.runner.policy, timestep_limit=test_tasks.spec.timestep_limit)
                logger.log(epoch_ind, post_update_rewards=np.mean(p['rewards']))
                comet_logger.log_metric(post_update_rewards, np.mean(p['rewards']), step=epoch_ind)
                maml.load_checkpoint()

            if should_plot and callable(plot_fn):
                plot_fn(save=True if should_save or is_the_end else False, lr=beta_lr)


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
