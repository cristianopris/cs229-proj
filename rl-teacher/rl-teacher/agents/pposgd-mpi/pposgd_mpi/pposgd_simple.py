import time
from collections import deque
from copy import deepcopy
from datetime import datetime
import os

import numpy as np
import pposgd_mpi.common.tf_util as U
import tensorflow as tf
from mpi4py import MPI
from pposgd_mpi.common import Dataset, explained_variance, fmt_row, zipsame
from pposgd_mpi.common import logger
from pposgd_mpi.common.mpi_adam import MpiAdam
from pposgd_mpi.common.mpi_moments import mpi_moments
from tensorflow.python.tools import freeze_graph


def traj_segment_generator(pi, env, steps_per_batch, stochastic, predictor=None):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    _ = env.reset()
    ob, rew, new, info = env.step(ac)  # Take one step so that we can get the datatype of info.get("human_obs")
    new = True  # marks if we're on first timestep of an episode

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(steps_per_batch)])
    human_obs = np.array([info.get("human_obs") for _ in range(steps_per_batch)])
    rews = np.zeros(steps_per_batch, 'float32')
    vpreds = np.zeros(steps_per_batch, 'float32')
    news = np.zeros(steps_per_batch, 'int32')
    acs = np.array([ac for _ in range(steps_per_batch)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        # t = global timestep number. never reset.
        if t > 0 and t % steps_per_batch == 0:
            path = {"obs": obs, "rew": rews, "vpred": vpreds, "new": news,
                    "actions": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets": ep_rets, "ep_lens": ep_lens, "human_obs": human_obs}
            # logger.log('path: ' + str(path))

            ################################
            #  START REWARD MODIFICATIONS  #
            ################################
            if predictor:
                path["original_rewards"] = path["rew"]
                logger.log('Predicting reward...')
                path["rew"] = predictor.predict_reward(path)
                last_callback = {}
                for ep_path in split_path_by_episode(path):
                    last_callback = predictor.path_callback(ep_path)
                    if (last_callback):
                        print('Predictor training info:', last_callback)
                if (hasattr(predictor, 'stat_sampled_segments')):
                    print("### Predictor segment stats: rejected : %d , sampled: %d , comparisons: %d" % (predictor.stat_rejected_segments, predictor.stat_sampled_segments, predictor.stat_total_comparisons))
                    # logger.log('Done reward modification')
            ################################
            #   END REWARD MODIFICATIONS   #
            ################################

            yield path

            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        i = t % steps_per_batch  # step count in this batch
        obs[i] = ob
        human_obs[i] = info.get("human_obs")
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, info = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew  # episode reward (original)
        cur_ep_len += 1  # episode length
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def split_path_by_episode(path):
    """Split path into episodes and yield a deepcopy of each one"""
    ep_breaks = np.where(path['new'])[0]
    start = ep_breaks[0]
    for end in ep_breaks[1:]:
        yield deepcopy({k: v[start:end] for k, v in path.items()
                        if k in ['obs', 'actions', 'original_rewards', 'pred_rewards', 'reshaper_rewards', 'human_obs']})
        start = end


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def learn(env, policy_func, *,
          timesteps_per_batch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0, save_freq=50000,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          predictor=None,
          env_name="env",
          load_checkpoint=False,
          experiment_name=None
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space)  # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                            shape=[])  # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="obs")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold
    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #
    pol_surr = - U.mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    vfloss1 = tf.square(pi.vpred - ret)
    vpredclipped = oldpi.vpred + tf.clip_by_value(pi.vpred - oldpi.vpred, -clip_param, clip_param)
    vfloss2 = tf.square(vpredclipped - ret)
    vf_loss = .5 * U.mean(
        tf.maximum(vfloss1, vfloss2))  # we do the same clipping-based trust region for the value function
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    saver = tf.train.Saver(pi.get_variables())

    if (load_checkpoint):
        load_model(U.get_session(), saver, env_name, checkpoint_subdir=load_checkpoint)
    else:
        U.initialize()
    adam.sync()

    #summaryWriter = tf.summary.FileWriter('summaries', U.get_session().graph)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, predictor=predictor)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        ##TODO
        logger.log("********** Iteration %i ************" % iters_so_far)

        if (iters_so_far % 5 == 0 and iters_so_far >= 1):
            if (predictor):
                predictor.save_model(_model_dir(env_name, experiment_name, timesteps_so_far))
            export_model(U.get_session(), saver=saver, env_name=env_name, experiment_name=experiment_name,
                         target_nodes='pi/action', steps=timesteps_so_far)

        logger.log("Generating segments...")
        seg = seg_gen.__next__()
        logger.log("Done generating segments")

        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["obs"], seg["actions"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before update
        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        d = Dataset(dict(obs=ob, actions=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        assign_old_eq_new()  # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["obs"], batch["actions"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["obs"], batch["actions"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


def load_model(sess, saver, env_name, checkpoint_subdir):
    dir = env_name + '_model' + "/" + checkpoint_subdir

    # print('Loading model from: ' + checkpoint_file)
    ckpt = tf.train.get_checkpoint_state(dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Done loading model...')

def _model_dir(env_name, experiment_name, steps):
    exp_name = (experiment_name if (experiment_name) else slugify(env_name + '_' + str(datetime.datetime.now())))
    dir = env_name + '_model' + '/' + exp_name + '/' + str(steps)
    os.makedirs(dir, exist_ok=True)
    return dir


def export_model(sess, saver, env_name, experiment_name, target_nodes, steps):
    dir = _model_dir(env_name, experiment_name, steps)

    checkpoint_file = dir + '/session-' + str(steps) + '.checkpoint'

    print("Saving checkpoint", checkpoint_file)
    saver.save(sess, checkpoint_file)

    model_protobuf_file = 'pposgd_policy_graph.pb'
    tf.train.write_graph(sess.graph_def, dir, model_protobuf_file, as_text=False)

    """
    Exports latest saved model to .bytes format for Unity embedding.
    :param model_path: path of model checkpoints.
    :param env_name: Name of associated Learning Environment.
    :param target_nodes: Comma separated string of needed output nodes for embedded graph.
    """
    print("Exporting model...")
    with tf.Session():
        exp_name = (experiment_name if (experiment_name) else slugify(env_name + '_' + str(datetime.datetime.now())))
        out_file_name = dir + '/' + exp_name + '_' + str(steps) + '.bytes'
        ckpt = tf.train.get_checkpoint_state(dir)
        freeze_graph.freeze_graph(input_graph=dir + '/' + model_protobuf_file,
                                  input_binary=True,
                                  input_checkpoint=ckpt.model_checkpoint_path,
                                  output_node_names=target_nodes,
                                  output_graph=out_file_name,
                                  clear_devices=True, initializer_nodes="", input_saver="",
                                  restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")
    print("Done exporting model: ", out_file_name)
