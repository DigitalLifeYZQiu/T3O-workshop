import argparse
from argparse import Namespace

parser = argparse.ArgumentParser(description='Hyperparameter tuning for time-series forecasting')
parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
parser.add_argument('--pred_len', type=int, required=True, help='Length of prediction')
parser.add_argument('--res_root_dir', type=str, required=True, help='Directory to save results')
parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU')
parser.add_argument('--gpu_indexes', type=str, default='', help='Indexes of GPUs to use')
# override fastmode
parser.add_argument('--fast_mode', action='store_true', help='Whether to use fast mode')
# seed
parser.add_argument('--seed', type=int, default=0, help='Random seed')  # FIXME: 0
# num_params num_samples ablation
parser.add_argument('--num_params', type=int, default=0, help='Number of parameters to try')
parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to try')
parser.add_argument('--ablation', type=str, default='none', help='Ablation study type')

args = parser.parse_args()

# FIXME:
# if args.fast_mode:
#     args = argparse.Namespace(data_name='ETTh1', model_name='Chronos-tiny', pred_len=720, res_root_dir='./debug',
#                               use_gpu=True, gpu_indexes='0', fast_mode=True)
# python3 ./AnyTransform/exp_single.py --data_name ETTh1 --model_name Chronos-tiny --pred_len 720 --res_root_dir ./debug --use_gpu --gpu_indexes 0 --fast_mode
# python3 ./AnyTransform/exp_single.py --data_name ETTh1 --model_name Chronos-tiny --pred_len 720 --res_root_dir ./debug --fast_mode
# CUDA_VISIBLE_DEVICES='1' python3 ./AnyTransform/exp_single.py --res_root_dir ./debug --use_gpu --gpu_indexes 1 --data_name Electricity --model_name Timer-UTSD --pred_len 96 >./debug/exp.log
# CUDA_VISIBLE_DEVICES='1' python3 ./AnyTransform/exp_single.py --res_root_dir ./debug --use_gpu --gpu_indexes 1 --data_name Electricity --model_name Timer-UTSD --pred_len 96  --num_params 25 --num_samples 100 --ablation none --seed 3 >./debug/exp.log
# 我希望他在判断os是mac的时候fast_mode=True
# fast_mode = True if sys.platform == 'darwin' else False

model_name = args.model_name
data_name = args.data_name
pred_len = args.pred_len
use_gpu = args.use_gpu
gpu_indexes = args.gpu_indexes
res_root_dir = args.res_root_dir
fast_mode = args.fast_mode
seed = args.seed
num_params = args.num_params
num_samples = args.num_samples
ablation = args.ablation

# patch_len = 96
# nan_inf_clip_factor = 5
