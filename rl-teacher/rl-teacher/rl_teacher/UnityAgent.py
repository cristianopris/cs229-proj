import numpy as np
import os
import tensorflow as tf
import gym

from ppo.history import *
from ppo.models import *
from ppo.trainer import Trainer
from unityagents import *

from rl_teacher.envs import TransparentWrapper
from rl_teacher.utils import *
from gym.spaces import Box

def train_unity_ppo(env_name, predictor):

    env_name = env_name  # Name of the training environment file.

    #os.chdir('bin')

    # if env is not None:
    #     env.close()

    env = UnityEnvironment(file_name=env_name, worker_id=os.getpid())
    print(str(env))

    brain_name = env.brain_names[0]

    max_steps = 5e5  # Set maximum number of steps to run environment.
    run_path = "unity_ppo"  # The sub-directory name for model and summary statistics
    load_model = False  # Whether to load a saved model.
    train_model = True  # Whether to train the model.
    summary_freq = 10000  # Frequency at which to save training statistics.
    save_freq = 50000  # Frequency at which to save model.

    ### Algorithm-specific parameters for tuning
    gamma = 0.99  # Reward discount rate.
    lambd = 0.95  # Lambda parameter for GAE.
    time_horizon = 2048  # How many steps to collect per agent before adding to buffer.
    beta = 1e-3  # Strength of entropy regularization
    num_epoch = 5  # Number of gradient descent steps per batch of experiences.
    epsilon = 0.2  # Acceptable threshold around ratio of old and new policy probabilities.
    buffer_size = 2048  # How large the experience buffer should be before gradient descent.
    learning_rate = 3e-4  # Model learning rate.
    hidden_units = 64  # Number of units in hidden layer.
    batch_size = 64  # How many experiences per gradient descent update step.

    tf.reset_default_graph()

    # Create the Tensorflow model graph
    ppo_model = create_agent_model(env, lr=learning_rate,
                                   h_size=hidden_units, epsilon=epsilon,
                                   beta=beta, max_step=max_steps)

    is_continuous = (env.brains[brain_name].action_space_type == "continuous")
    use_observations = (env.brains[brain_name].number_observations > 0)
    use_states = (env.brains[brain_name].state_space_size > 0)

    model_path = './models/{}'.format(run_path)
    summary_path = './summaries/{}'.format(run_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, \
                            allow_soft_placement=True, device_count={'CPU': 4})
    with tf.Session(config=config) as sess:
        # Instantiate model parameters
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)
        steps = sess.run(ppo_model.global_step)
        summary_writer = tf.summary.FileWriter(summary_path)
        info = env.reset(train_mode=train_model)[brain_name]
        trainer = Trainer(ppo_model, sess, info, is_continuous, use_observations, use_states)
        while steps <= max_steps:
            if env.global_done:
                info = env.reset(train_mode=train_model)[brain_name]
            # Decide and take an action
            new_info = trainer.take_action(info, env, brain_name)
            info = new_info
            trainer.process_experiences(info, time_horizon, gamma, lambd)
            if len(trainer.training_buffer['actions']) > buffer_size and train_model:
                # Perform gradient descent with experience buffer
                trainer.update_model(batch_size, num_epoch)
            if steps % summary_freq == 0 and steps != 0 and train_model:
                # Write training statistics to tensorboard.
                trainer.write_summary(summary_writer, steps)
            if steps % save_freq == 0  and train_model:
                # Save Tensorflow model
                save_model(sess, model_path=model_path, steps=steps, saver=saver)
                export_graph(model_path, env_name = env_name + '_steps' + str(steps))
            steps += 1
            sess.run(ppo_model.increment_step)
        # Final save Tensorflow model
        if steps != 0 and train_model:
            save_model(sess, model_path=model_path, steps=steps, saver=saver)
            export_graph(model_path, env_name & '_steps' & str(steps))
    env.close()

env_instances = {}

def make_unity_env(env_id):
    env_name = env_id[6:] if env_id.startswith('unity-') else env_id
    env = env_instances.get(env_name)
    print(env_instances)
    if env is None:
        env = UnityGymWrapper(env_name)
        env_instances[env_name] = env
    return env

class UnityGymWrapper(gym.Env):

    def __init__(self, file_name, worker_id= (os.getpid() % 13),
                 base_port=5005):
        self.env = UnityEnvironment(file_name, worker_id=worker_id, base_port=base_port)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
#        self.spec = False

        self.fps = 240
        self._max_episode_steps = 50
        n_agents = 1

        self.action_space = Box(np.tile([-15.0 , -15.0], n_agents), np.tile([15.0 , 15.0], n_agents))

        #state space:
        # platform rotation (z, x)
        # relative ball position (x , y, z) /5f
        # ball velocity (x, y, z) /5f
        box_low =  np.array([-1., -1.,  -3., -3., -3.,  -2.,  -2. , -2. ])
        box_high = np.array([ 1. , 1.,   3.,  3.,  3.,   2.,   2. ,  2. ])
        self.observation_space = Box(np.tile(box_low, n_agents), np.tile(box_high, n_agents))

        print('UnityGymWrapper: created env: ', self.env)

    def _step(self, action):
        brain_info = self.env.step(action)[self.brain_name]
        n_agents = len(brain_info.agents)
        state = brain_info.states.reshape(8 * n_agents)

        #print('state:', state)
        #print('action:', action)
        frames = None
        #if (brain_info.observations):
        frames  = brain_info.observations[0][0] #brain, camera

        return state, brain_info.rewards[0], brain_info.local_done[0], {'human_obs' : frames}
        """
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """


    def _reset(self):
        brain_info = self.env.reset()[self.brain_name]
        n_agents = len(brain_info.agents)
        state = brain_info.states.reshape(8 * n_agents)
        return state

    def _render(self, mode='human', close=False): return

    def _seed(self, seed=None): return []


def train_unity_pposgd_mpi(env_name, make_env, num_timesteps, seed, experiment_name, predictor=None, load_checkpoint=False):
    from pposgd_mpi import mlp_policy
    from pposgd_mpi import pposgd_simple
    from pposgd_mpi import bench
    from pposgd_mpi.common import logger
    from pposgd_mpi.common import set_global_seeds, tf_util as U
    import logging




    U.make_session(num_cpu=6).__enter__()
    logger.session(dir=model_dir(env_name, experiment_name)).__enter__()
    set_global_seeds(seed)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    env = make_env(env_name)

    env = bench.Monitor(env, os.path.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
        max_timesteps=num_timesteps,
        timesteps_per_batch=128*10,
        clip_param=0.2, entcoeff=0.0,
        optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        predictor=predictor,
        env_name=env_name,
        load_checkpoint=load_checkpoint,
        experiment_name=experiment_name
        )
    env.close()