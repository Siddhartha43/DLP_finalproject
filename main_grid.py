# -*-Encoding: utf-8 -*-
"""
Description:
    If you use any part of the code in this repository, please consider citing the following paper:
    Yan Li et al. Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement,
    in Proceedings of 36th Conference on Neural Information Processing Systems (NeurIPS '22),
    November 28 -- December 9, 2022.
Authors:
    Li,Yan (liyan22021121@gmail.com)
"""
import argparse
import paddle
import numpy as np
import random
import csv
from exp.exp_model import Exp_Model

fix_seed = 1
random.seed(fix_seed)
# paddle.seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='generating')

parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')  # change it accordingly
parser.add_argument('--target', type=str, default='target', help='target variable in S or MS task')
parser.add_argument('--features', type=str, default='MS', help='forecasting task, options: [M, S, MS]; '
                                                               'M: multivariate predict multivariate, '
                                                               'S: univariate predict univariate, '
                                                               'MS: multivariate predict univariate')
parser.add_argument('--freq', type=str, default='t', help='frequency for time features encoding, '
                                                          'options: [s -- second, t -- minutely, h -- hourly, '
                                                          'd -- daily, b -- business days, w -- week, m -- month], '
                                                          'you can also use more detailed frequency like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--arch_instance', type=str, default='res_mbconv', help='the architecture type')

# load data
parser.add_argument('--sequence_length', type=int, default=16, help='input sequence length')
parser.add_argument('--prediction_length', type=int, default=16, help='prediction sequence length')
parser.add_argument('--percentage', type=float, default=0.05, help='the used percentage of the whole dataset')
parser.add_argument('--target_dim', type=int, default=1, help='dimension of target')
parser.add_argument('--input_dim', type=int, default=321, help='dimension of input')
parser.add_argument('--hidden_size', type=int, default=128, help='dimension of the hidden sstates')
parser.add_argument('--embedding_dimension', type=int, default=64, help='feature embedding dimension')

# diffusion process
parser.add_argument('--diff_steps', type=int, default=100, help='number of the diff step')
parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
parser.add_argument('--beta_schedule', type=str, default='linear', help='the type of beta schedule')
parser.add_argument('--beta_start', type=float, default=0.0, help='start of the beta schedule')
parser.add_argument('--beta_end', type=float, default=0.01, help='end of the beta schedule')
parser.add_argument('--scale', type=float, default=0.1, help='set smaller diffusion scale for target')

parser.add_argument('--psi', type=float, default=0.5, help='trade off parameter psi')
parser.add_argument('--lambda1', type=float, default= 1, help='trade off parameter lambda')
parser.add_argument('--gamma', type=float, default=0.01, help='trade off parameter gamma')

# Bidirectional VAE
parser.add_argument('--mult', type=float, default=1, help='mult of channels')
parser.add_argument('--num_layers', type=int, default=2, help='num of RNN layers')
parser.add_argument('--num_channels_enc', type=int, default=32, help='initinal number of channels in encoder')
parser.add_argument('--channel_mult', type=int, default=2, help='multiplier of the channels')
parser.add_argument('--num_preprocess_blocks', type=int, default=1, help='number of preprocessing blocks')
parser.add_argument('--num_preprocess_cells', type=int, default=3, help='number of cells per block')
parser.add_argument('--groups_per_scale', type=int, default=2, help='number of groups')
parser.add_argument('--num_postprocess_blocks', type=int, default=1, help='number of postprocessing blocks')
parser.add_argument('--num_postprocess_cells', type=int, default=2, help='number of cells per block')
parser.add_argument('--num_channels_dec', type=int, default=32, help='number of channels in decoder')
parser.add_argument('--num_latent_per_group', type=int, default=8, help='number of channels in latent variables per group')

# training settings
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--patience', type=int, default=5, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--dim', type=int, default=-1, help='forecasting dims')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay')
parser.add_argument('--loss_type', type=str, default='kl',help='loss function')
parser.add_argument('--inverse', type=bool, default=False,help='whether inverse the output')

# device
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=3, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multiple gpus')

args = parser.parse_args()
args.use_gpu = True if paddle.device.is_compiled_with_cuda() and args.use_gpu else False

# 定義超參數網格
beta_end_values = [0.01, 0.05, 0.1]
diff_steps_values = [50, 100, 1000, 2000]
beta_schedule_values = ['linear', 'quad']  # <-- Modified: Add potential beta_schedule values for grid search

# 創建一個列表來保存每組超參數的結果
results = []

# 網格搜索
for beta_end_val in beta_end_values:
    for diff_steps_val in diff_steps_values:
        for beta_schedule_val in beta_schedule_values:
            # 設定超參數
            args.beta_end = beta_end_val
            args.diff_steps = diff_steps_val
            args.beta_schedule = beta_schedule_val  
            
            # 顯示當前使用的超參數
            print(f"Training with beta_end: {args.beta_end}, diff_steps: {args.diff_steps}, beta_schedule: {args.beta_schedule}")
            
            all_mse = []
            all_mae = []
            for ii in range(0, args.itr):
                setting = '{}_sl_{}_pl{}_{}_dim{}_scale{}_diffsteps{}_betaend{}_betaschedule{}_itr{}'.format(args.data_path, 
                            args.sequence_length, args.prediction_length, ii, args.dim, args.scale, 
                            args.diff_steps, args.beta_end, args.beta_schedule, ii)
                
                exp = Exp_Model(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
                print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                mae, mse = exp.test(setting)
                all_mae.append(mae)
                all_mse.append(mse)
                paddle.device.cuda.empty_cache()

            # 計算MAE和MSE的平均值
            avg_mae = np.mean(all_mae)
            avg_mse = np.mean(all_mse)
            
            # 將結果添加到results列表中
            results.append((beta_end_val, diff_steps_val, beta_schedule_val, avg_mse, avg_mae))


# 寫入CSV文件
with open('./results/grid_search_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["beta_end", "diff_steps", "beta_schedule", "avg_mse", "avg_mae"])  
    for result in results:
        writer.writerow(result)

print("Grid search results saved to grid_search_results.csv")

# 找出最佳的超參數組合
best_params = min(results, key=lambda x: x[3])  # 以MSE為基準
print(f"Best parameters (beta_end, diff_steps, beta_schedule): ({best_params[0]}, {best_params[1]}, {best_params[2]}) with MSE: {best_params[3]} and MAE: {best_params[4]}") 

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

all_mse = []
all_mae = []
for ii in range(0, args.itr):

    setting = '{}_sl_{}_pl{}_{}_dim{}_scale{}_diffsteps{}_itr{}'.format(args.data_path, args.sequence_length,
             args.prediction_length, ii, args.dim,  args.scale, args.diff_steps, ii)
    exp = Exp_Model(args)  # set experiments

    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    mae, mse = exp.test(setting)
    all_mae.append(mae)
    all_mse.append(mse)
    paddle.device.cuda.empty_cache()

print(np.mean(np.array(all_mse)), np.std(np.array(all_mse)),
      np.mean(np.array(all_mae)), np.std(np.array(all_mae)))
