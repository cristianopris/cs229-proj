import os
import os.path as osp
import random
from collections import deque
from time import time, sleep

import numpy as np

from parallel_trpo.train import train_parallel_trpo
from pposgd_mpi.run_mujoco import train_pposgd_mpi

from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_with_torque_removed
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.nn import FullyConnectedMLP
from rl_teacher.segment_sampling import sample_segment_from_path
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify, corrcoef
from rl_teacher.video import SegmentVideoRecorder
from rl_teacher.predictor import *

from UnityAgent import *



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', required=True)
    parser.add_argument('-p', '--predictor', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=1, type=int)
    parser.add_argument('-l', '--n_labels', default=None, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=5e6, type=int)
    parser.add_argument('-a', '--agent', default="parallel_trpo", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=10000, type=int)
    parser.add_argument('-V', '--no_videos', action="store_true")
    args = parser.parse_args()

    print("Setting things up...")

    env_id = args.env_id
    run_name = "%s/%s-%s" % (env_id, args.name, int(time()))
    summary_writer = make_summary_writer(run_name)


    make_env = None
    env = None
    if env_id.startswith('unity-'):
        env_name = env_id[6:]
        env = make_unity_env(env_name)
        make_env = make_unity_env
    else:
        env = make_with_torque_removed(env_id)

    num_timesteps = int(args.num_timesteps)
    experiment_name = slugify(args.name)

    if args.predictor == "rl":
        predictor = TraditionalRLRewardPredictor(summary_writer)
    else:
        agent_logger = AgentLogger(summary_writer)

        pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels // 4

        if args.n_labels:
            label_schedule = LabelAnnealer(
                agent_logger,
                final_timesteps=num_timesteps,
                final_labels=args.n_labels,
                pretrain_labels=pretrain_labels)
        else:
            print("No label limit given. We will request one label every few seconds.")
            label_schedule = ConstantLabelSchedule(pretrain_labels=pretrain_labels)

        if args.predictor == "synth":
            comparison_collector = SyntheticComparisonCollector()

        elif args.predictor == "human":
            #bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
            #assert bucket and bucket.startswith("gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"
            comparison_collector = HumanComparisonCollector(env_id, env = env, experiment_name=experiment_name)
        else:
            raise ValueError("Bad value for --predictor: %s" % args.predictor)

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(
            env_id, make_env, n_desired_segments=pretrain_labels * 2,
            clip_length_in_seconds=CLIP_LENGTH, workers=args.workers)
        for i in range(pretrain_labels):  # Turn our random segments into comparisons
            comparison_collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + pretrain_labels])

        # Sleep until the human has labeled most of the pretraining comparisons
        while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
            comparison_collector.label_unlabeled_comparisons()
            if args.predictor == "synth":
                print("%s synthetic labels generated... " % (len(comparison_collector.labeled_comparisons)))
            elif args.predictor == "human":
                print("%s/%s comparisons labeled. Please add labels w/ the human-feedback-api. Sleeping... " % (
                    len(comparison_collector.labeled_comparisons), pretrain_labels))
                sleep(5)

        predictor = ComparisonRewardPredictor(
            env,
            summary_writer,
            comparison_collector=comparison_collector,
            agent_logger=agent_logger,
            label_schedule=label_schedule,
        )

        # Start the actual training
        for i in range(args.pretrain_iters):
            predictor.train_predictor()  # Train on pretraining labels
            if i % 100 == 0:
                print("%s/%s predictor pretraining iters... " % (i, args.pretrain_iters))

    # Wrap the predictor to capture videos every so often:
    if not args.no_videos:
        predictor = SegmentVideoRecorder(predictor, env, save_dir=osp.join('/tmp/rl_teacher_vids', run_name))

    # We use a vanilla agent from openai/baselines that contains a single change that blinds it to the true reward
    # The single changed section is in `rl_teacher/agent/trpo/core.py`
    print("Starting joint training of predictor and agent")
    if args.agent == "parallel_trpo":
        train_parallel_trpo(
            env_id=env_id,
            make_env=make_with_torque_removed,
            predictor=predictor,
            summary_writer=summary_writer,
            workers=args.workers,
            runtime=(num_timesteps / 1000),
            max_timesteps_per_episode=get_timesteps_per_episode(env),
            timesteps_per_batch=8000,
            max_kl=0.001,
            seed=args.seed,
        )
    elif args.agent == "pposgd_mpi":
        def make_env():
            return make_with_torque_removed(env_id)
        train_pposgd_mpi(make_env, num_timesteps=num_timesteps, seed=args.seed, predictor=predictor)
    elif args.agent == "unity-ppo":
        env_name = env_id[6:] if env_id.startswith('unity-') else env_id
        train_unity_ppo(env_name=env_name, predictor=predictor)
    elif args.agent == "unity-pposgd-mpi":
        env_name = env_id[6:] if env_id.startswith('unity-') else env_id # remove unity- prefix
        train_unity_pposgd_mpi(env_name=env_name, make_env=make_env, num_timesteps=num_timesteps, seed=args.seed, predictor=predictor)
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()
