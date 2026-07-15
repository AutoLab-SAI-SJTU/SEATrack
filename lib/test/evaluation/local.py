from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/got10k_lmdb'
    settings.got10k_path = '/home/yufan/code/SEATrack/datasets/GOT-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/itb'
    settings.lasot_extension_subset_path_path = '/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/lasot_lmdb'
    settings.lasot_path = '/home/yufan/code/SEATrack/datasets/LaSOT/LaSOTBenchmark'
    settings.network_path = '/home/yufan/code/SEATrack-ProbAlign-VRE/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/nfs'
    settings.otb_path = '/data/otb'
    settings.prj_dir = '/home/yufan/code/SEATrack-ProbAlign-VRE'
    settings.result_plot_path = '/home/yufan/code/SEATrack-ProbAlign-VRE/output/test/result_plots'
    settings.results_path = '/home/yufan/code/SEATrack-ProbAlign-VRE/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/yufan/code/SEATrack-ProbAlign-VRE/output'
    settings.segmentation_path = '/home/yufan/code/SEATrack-ProbAlign-VRE/output/test/segmentation_results'
    settings.tc128_path = '/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/yufan/code/SEATrack/datasets/TNL2K'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/yufan/code/SEATrack/datasets/TrackingNet'
    settings.uav_path = '/home/yufan/code/SEATrack/datasets/UAV123'
    settings.vot18_path = '/data/vot2018'
    settings.vot22_path = '/home/yufan/code/SEATrack/datasets/VOT22'
    settings.vot_path = '/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings
