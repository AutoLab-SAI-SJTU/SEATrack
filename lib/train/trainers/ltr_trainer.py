import os
import datetime
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.amp import autocast, GradScaler
    _TORCH_AMP_REQUIRES_DEVICE = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _TORCH_AMP_REQUIRES_DEVICE = False
from lib.utils.misc import get_world_size
from tqdm.auto import tqdm
from lib.train.admin.logging_utils import get_train_logger


class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        # if use_amp:
        if _TORCH_AMP_REQUIRES_DEVICE:
            self.scaler = GradScaler("cuda", enabled=use_amp)
        else:
            self.scaler = GradScaler(enabled=use_amp)
        self.train_logger = get_train_logger(self.settings, "train")

    def _show_progress(self):
        return getattr(self.settings, "local_rank", -1) in (-1, 0)

    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        '''add fix rgb pretrained net bn, only used in box_head'''
        if self.settings.fix_bn:
            self.actor.fix_bns()

        self._init_timing()

        batch_size = 0
        progress_desc = "%s epoch %d" % (loader.name, self.epoch)
        progress_iter = enumerate(loader, 1)
        with tqdm(progress_iter, total=len(loader), desc=progress_desc,
                  dynamic_ncols=True, disable=not self._show_progress()) as progress:
            for i, data in progress:
                self.data_read_done_time = time.time()
                # get inputs
                if self.move_data_to_gpu:
                    data = data.to(self.device)

                self.data_to_gpu_time = time.time()

                data['epoch'] = self.epoch
                data['settings'] = self.settings
                # forward pass
                # if not self.use_amp:
                #     loss, stats = self.actor(data)
                # else:
                if _TORCH_AMP_REQUIRES_DEVICE:
                    amp_context = autocast("cuda", enabled=self.use_amp)
                else:
                    amp_context = autocast(enabled=self.use_amp)
                with amp_context:
                    loss, stats = self.actor(data)

                # backward pass and update weights
                if loader.training:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    if self.settings.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # update statistics
                batch_size = data['template_images'].shape[loader.stack_dim]
                self._update_stats(stats, batch_size, loader)

                # print statistics
                self._print_stats(i, loader, batch_size, progress)

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        if self.num_frames > 0:
            self.train_logger.info(
                "Epoch %d %s summary: epoch_time=%s avg_data_time=%.5f avg_gpu_trans_time=%.5f avg_forward_time=%.5f",
                self.epoch, loader.name, datetime.timedelta(seconds=epoch_time),
                self.avg_date_time / self.num_frames * batch_size,
                self.avg_gpu_trans_time / self.num_frames * batch_size,
                self.avg_forward_time / self.num_frames * batch_size,
            )

    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                if hasattr(loader, "set_epoch"):
                    loader.set_epoch(self.epoch)
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size, progress=None):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if progress is not None:
            progress.set_postfix(self._build_progress_postfix(loader, average_fps, batch_fps, batch_size),
                                 refresh=False)

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)
            # print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.data_read_done_time - prev_frame_time_backup, self.data_to_gpu_time - self.data_read_done_time)
            # print_str += 'ForwardTime: %.3f  ,  ' % (current_time - self.data_to_gpu_time)
            # print_str += 'TotalTime: %.3f  ,  ' % (current_time - prev_frame_time_backup)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)
                    # else:
                    #     print_str += '%s: %r  ,  ' % (name, val)

            self.train_logger.info(print_str[:-5])

    def _build_progress_postfix(self, loader, average_fps, batch_fps, batch_size):
        postfix = {
            "fps": "%.1f" % average_fps,
            "batch_fps": "%.1f" % batch_fps,
            "data": "%.3f" % (self.avg_date_time / self.num_frames * batch_size),
            "fwd": "%.3f" % (self.avg_forward_time / self.num_frames * batch_size),
        }
        loader_stats = self.stats.get(loader.name)
        if loader_stats is not None:
            for name in ("Loss/total", "Loss/giou", "Loss/l1", "IoU"):
                val = loader_stats.get(name)
                if hasattr(val, "avg"):
                    postfix[name.split("/")[-1]] = "%.4f" % val.avg
        return postfix

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
