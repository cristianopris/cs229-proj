import multiprocessing
import os
import os.path as osp
import uuid

import numpy as np
from copy import deepcopy

from rl_teacher.envs import make_with_torque_removed
from rl_teacher.video import write_segment_to_video, upload_to_gcs, export_video
from rl_teacher.utils import *


class SyntheticComparisonCollector(object):
    def __init__(self):
        self._comparisons = []

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "label": None
        }
        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        for comp in self.unlabeled_comparisons:
            self._add_synthetic_label(comp)

    @staticmethod
    def _add_synthetic_label(comparison):
        left_seg = comparison['left']
        right_seg = comparison['right']
        left_has_more_rew = np.sum(left_seg["original_rewards"]) > np.sum(right_seg["original_rewards"])

        # Mutate the comparison and give it the new label
        comparison['label'] = 0 if left_has_more_rew else 1

# def _write_and_upload_video(env_id, gcs_path, local_path, segment):
#     # env = make_with_torque_removed(env_id)
#     # write_segment_to_video(segment, fname=local_path, env=env)
#     # upload_to_gcs(local_path, gcs_path)

class HumanComparisonCollector():

    def __init__(self, env_id, fps, experiment_name):
        from human_feedback_api import Comparison

        self._comparisons = []
        self.env_id = env_id
        self.experiment_name = experiment_name
        self._upload_workers = multiprocessing.Pool(4)
        self.fps = fps

        logdir = model_dir(env_id, experiment_name)
        self.logger = JSONLogger(logdir + '/comparisons.json')

        if Comparison.objects.filter(experiment_name=experiment_name).count() > 0:
            raise EnvironmentError("Existing experiment named %s! Pick a new experiment name." % experiment_name)

    # def _orig_convert_segment_to_media_url(self, comparison_uuid, side, segment):
    #     tmp_media_dir = '/tmp/rl_teacher_media'
    #     media_id = "%s-%s.mp4" % (comparison_uuid, side)
    #     local_path = osp.join(tmp_media_dir, media_id)
    #     gcs_bucket = os.environ.get('RL_TEACHER_GCS_BUCKET')
    #     gcs_path = osp.join(gcs_bucket, media_id)
    #     self._upload_workers.apply_async(_write_and_upload_video, (self.env_id, gcs_path, local_path, segment))
    #
    #     media_url = "https://storage.googleapis.com/%s/%s" % (gcs_bucket.lstrip("gs://"), media_id)
    #     return media_url

    def convert_segment_to_media_url(self, comparison_uuid, side, segment):
        tmp_media_dir = 'rl_teacher_media'
        media_id = "%s-%s.mp4" % (comparison_uuid, side)
        local_path = osp.join(tmp_media_dir, media_id)
        media_url = '/media/%s' % (media_id)

        os.makedirs(osp.dirname(local_path), exist_ok=True)

        frames = [f for f in segment["human_obs"]]

        # for i in range(int(self.fps * 0.2)):
        #     frames.append(frames[-1])

        export_video(frames, local_path, fps=self.fps)
        return media_url


    def _create_comparison_in_webapp(self, left_seg, right_seg):
        """Creates a comparison DB object. Returns the db_id of the comparison"""
        from human_feedback_api import Comparison

        comparison_uuid = str(uuid.uuid4())
        comparison = Comparison(
            experiment_name=self.experiment_name,
            media_url_1=self.convert_segment_to_media_url(comparison_uuid, 'left', left_seg),
            media_url_2=self.convert_segment_to_media_url(comparison_uuid, 'right', right_seg),
            response_kind='left_or_right',
            priority=1.
        )
        comparison.full_clean()
        comparison.save()
        return comparison.id

    def add_segment_pair(self, left_seg, right_seg):
        """Add a new unlabeled comparison from a segment pair"""

        comparison_id = self._create_comparison_in_webapp(left_seg, right_seg)
        comparison = {
            "left": left_seg,
            "right": right_seg,
            "id": comparison_id,
            "label": None
        }

        self._comparisons.append(comparison)

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is not None]

    @property
    def labeled_decisive_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] in [0, 1]]

    @property
    def unlabeled_comparisons(self):
        return [comp for comp in self._comparisons if comp['label'] is None]

    def label_unlabeled_comparisons(self):
        from human_feedback_api import Comparison

        for comparison in self.unlabeled_comparisons:
            db_comp = Comparison.objects.get(pk=comparison['id'])
            if db_comp.response == 'left':
                comparison['label'] = 0
            elif db_comp.response == 'right':
                comparison['label'] = 1
            elif db_comp.response == 'tie' or db_comp.response == 'abstain':
                comparison['label'] = 'equal'
                # If we did not match, then there is no response yet, so we just wait
            if (comparison['label'] is not None):
                c = deepcopy(comparison)
                c['left']['human_obs'] = None
                c['right']['human_obs'] = None
                self.logger.writekvs(c)

