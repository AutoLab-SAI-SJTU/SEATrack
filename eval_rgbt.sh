# test lasher
CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name seatrack --dataset_name LasHeR --yaml_name rgbt --num_gpus 1

# test rgbt234
CUDA_VISIBLE_DEVICES=0 python ./RGBT_workspace/test_rgbt_mgpus.py --script_name seatrack --dataset_name RGBT234 --yaml_name rgbt --num_gpus 1
