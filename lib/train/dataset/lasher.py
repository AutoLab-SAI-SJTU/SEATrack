import os
import os.path
import numpy as np
import torch
import csv
import pandas
import glob
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class LasHeR(BaseVideoDataset):
    """ LasHeR dataset(aligned version).

    Publication:
        A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/pdf/2104.13202.pdf

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val', 'all', 'smoke'], 'Only support all, train, val, or smoke split in LasHeR, got {}'.format(split)
        super().__init__('LasHeR', root)
        self.dtype = dtype
        self._frame_start_cache = {}
        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        # seq_id is the index of the folder inside the got10k root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'lasher'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        file_path = os.path.join(ltr_path, 'data_specs', 'lasher_{}.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):
        # in lasher dataset, visible.txt is same as infrared.txt
        rgb_bb_anno_file = os.path.join(seq_path, "visible.txt")
        # ir_bb_anno_file = os.path.join(seq_path, "infrared.txt")
        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        # ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(rgb_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        """2022/8/10 ir and rgb have synchronous w=h=0 frame_index"""
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    @staticmethod
    def _modality_tokens(modality):
        if modality == 'visible':
            return ('', 'v', 'visible'), ('', 'v')
        return ('', 'i', 'infrared'), ('', 'i')

    @classmethod
    def _frame_name_candidates(cls, frame_id, modality):
        prefixes, suffixes = cls._modality_tokens(modality)

        candidates = []
        for prefix in prefixes:
            for suffix in suffixes:
                for width in (0, 4, 5, 6, 8):
                    number = str(frame_id) if width == 0 else '{:0{}d}'.format(frame_id, width)
                    candidates.append('{}{}{}.jpg'.format(prefix, number, suffix))
        return list(dict.fromkeys(candidates))

    @classmethod
    def _parse_frame_id(cls, frame_name, modality):
        stem, ext = os.path.splitext(frame_name)
        if ext.lower() != '.jpg':
            return None

        prefixes, suffixes = cls._modality_tokens(modality)
        for prefix in prefixes:
            if prefix and not stem.startswith(prefix):
                continue
            prefix_stripped = stem[len(prefix):] if prefix else stem
            for suffix in suffixes:
                if suffix and not prefix_stripped.endswith(suffix):
                    continue
                number = prefix_stripped[:-len(suffix)] if suffix else prefix_stripped
                if number.isdigit():
                    return int(number)
        return None

    def _get_frame_start_index(self, frame_dir, modality):
        cache_key = (frame_dir, modality)
        if cache_key not in self._frame_start_cache:
            frame_ids = [
                frame_id for frame_id in
                (self._parse_frame_id(frame_name, modality) for frame_name in os.listdir(frame_dir))
                if frame_id is not None
            ]
            self._frame_start_cache[cache_key] = min(frame_ids) if frame_ids else 0
        return self._frame_start_cache[cache_key]

    def _find_exact_frame(self, frame_dir, frame_id, modality):
        for frame_name in self._frame_name_candidates(frame_id, modality):
            frame_path = os.path.join(frame_dir, frame_name)
            if os.path.isfile(frame_path):
                return frame_path
        return None

    def _get_frame_path_for_modality(self, seq_path, subdir, frame_id, modality):
        frame_dir = os.path.join(seq_path, subdir)
        target_frame_id = frame_id + self._get_frame_start_index(frame_dir, modality)

        frame_path = self._find_exact_frame(frame_dir, target_frame_id, modality)
        if frame_path is not None:
            return frame_path

        pattern = '*{}.jpg'.format(target_frame_id)
        matches = sorted(glob.glob(os.path.join(frame_dir, pattern)))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            'Could not find {} frame {} in {}'.format(modality, target_frame_id, frame_dir)
        )

    def _get_frame_path(self, seq_path, frame_id):
        rgb_frame_path = self._get_frame_path_for_modality(seq_path, 'visible', frame_id, 'visible')
        ir_frame_path = self._get_frame_path_for_modality(seq_path, 'infrared', frame_id, 'infrared')
        return (rgb_frame_path, ir_frame_path)  # jpg jpg

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img  # (h,w,6)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
