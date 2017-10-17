import numpy as np
import os
import tensorflow as tf
import gym

from ppo.history import *
from ppo.models import *
from ppo.trainer import Trainer
from unityagents import *

from rl_teacher.envs import TransparentWrapper
from gym.spaces import Box

def train_unity_agent(env_id, predictor):
    ### Hyperparameters
    env_name = env_id  # Name of the training environment file.

    #os.chdir('bin')

    # if env is not None:
    #     env.close()

    env = UnityEnvironment(file_name=env_name)
    print(str(env))
    _train_model(env_name, env, predictor)


def _train_model(env_name, env, predictor):

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



class UnityGymWrapper(gym.Env):


    def __init__(self, file_name, worker_id=0,
                 base_port=5005):
        self.env = UnityEnvironment(file_name, worker_id=worker_id, base_port=base_port)
        self.brain_name = self.env.brain_names[0]
        self.spec = False

        self.action_space = Box(np.array([0.0 ,0.0]), np.array([359.0,359.0]))
        self.observation_space = Box(np.array([0. ,0., 0., 0., 0., 0., 0., 0.]), np.array([359.0,359.0, 100, 100, 100, 100, 100, 100]))

    def _step(self, action):
        brain_info = self.env.step(action)[self.brain_name]
        return brain_info.states[0], brain_info.rewards[0], brain_info.local_done[0], {} #info
        """
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """


    def _reset(self):
        brain_info = self.env.reset()[self.brain_name]
        return brain_info.observations

    def _render(self, mode='human', close=False): return

    def _seed(self, seed=None): return []

