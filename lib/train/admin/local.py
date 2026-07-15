class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/yufan/code/SEATrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/yufan/code/SEATrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/yufan/code/SEATrack/pretrained_networks'
        self.got10k_val_dir = '/home/yufan/code/SEATrack/datasets/GOT-10k/val'
        self.lasot_lmdb_dir = '/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data/coco_lmdb'
        self.coco_dir = '/data/coco'
        self.lasot_dir = '/home/yufan/code/SEATrack/datasets/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/home/yufan/code/SEATrack/datasets/GOT-10k/train'
        self.trackingnet_dir = '/home/yufan/code/SEATrack/datasets/TrackingNet'
        self.depthtrack_dir = '/home/yufan/code/SEATrack/datasets/DepthTrack/train'
        self.lasher_dir = '/home/yufan/code/SEATrack/datasets/LasHeR/trainingset'
        self.visevent_dir = '/home/yufan/code/SEATrack/datasets/VisEvent/train_subset'
