from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.seatrack.config import cfg, update_config_from_file


def parameters(yaml_name: str, epoch=None, variants=None):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/seatrack/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    if os.environ.get("SEATRACK_PRINT_CONFIG") == "1":
        print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    checkpoint_override = os.environ.get("SEATRACK_CHECKPOINT")
    if checkpoint_override:
        params.checkpoint = checkpoint_override
    else:
        # Network checkpoint path
        if yaml_name.startswith('rgbt'):
            # variants = os.environ['VARIANTS']
            # params.checkpoint = os.path.join(save_dir, f"checkpoints/{variants}/{epoch}.pth.tar")
            params.checkpoint = os.path.join(save_dir, "checkpoints/rgbt/SEATrack_ep0060.pth.tar")

        elif yaml_name.startswith('rgbd'):
            # variants = os.environ['VARIANTS']
            # params.checkpoint = os.path.join(save_dir, f"checkpoints/rgbd_random/SEATrack_{variants}_ep0025.pth.tar")
            params.checkpoint = os.path.join(save_dir, "checkpoints/rgbd/SEATrack_ep0025.pth.tar")

        elif yaml_name.startswith('rgbe'):
            # variants = os.environ['VARIANTS']
            # params.checkpoint = os.path.join(save_dir, f"checkpoints/{variants}/{epoch}.pth.tar")
            params.checkpoint = os.path.join(save_dir, "checkpoints/rgbe/SEATrack_ep0045.pth.tar")
        else:
            raise ValueError(f"Unsupported SEATrack yaml_name: {yaml_name}")

    # whether to save boxes from all queries
    params.save_all_boxes = False
    # params.debug = 1
    params.task = os.environ.get("SEATRACK_TASK", yaml_name)
    return params
