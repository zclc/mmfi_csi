import datetime
import os
import argparse
import sys
import subprocess

import yaml
import numpy as np
import torch

from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import cal_mpjpe

import datetime

from model import resnet34
import torch.optim as optim
from tqdm import tqdm
from mmfi_lib.evaluate import calulate_error
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("dataset_root", type=str, help="Root of Dataset")
    parser.add_argument("config_file", type=str, help="Configuration YAML file")
    args = parser.parse_args()

    log_dir_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    tb_writer = SummaryWriter(log_dir=os.path.join('logs', log_dir_name))
    dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # [17,3] [17, 3]
    if config['device'] == 'cpu':
        device = 'cpu'
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'using {device} device!')
    # torch.backends.cudnn.enabled = False

    net = resnet34()
    net.to(device)
    # model_weight_path = './resnet34.pth'
    # if not os.path.exists(model_weight_path):
    #     print("file {} does not exist.".format(model_weight_path))
    #     model_weight_path = '../03_ResNet/resnet34-pre.pth'
    #     print("using file {} pre-train.".format(model_weight_path))
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    if not config['weight_path']:
        model_weight_path = config['weight_path']
        print("using file {} pre-train.".format(model_weight_path))
        net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    loss_function = torch.nn.MSELoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    # Codes for training
    epochs = config['epochs']
    best_acc = 0.0
    pro_start_dt = datetime.datetime.now()
    for epoch in range(epochs):
        train_start_dt = datetime.datetime.now()
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout, leave=False)
        train_num = len(train_loader)
        for batch_idx, batch_data in enumerate(train_bar):
            #features = batch_data['input_wifi-csi']  # [bs, 1, 136, 136]
            features = batch_data['input_mmwave']
            labels = batch_data['output']  # [bs, 1, 17, 3]
            optimizer.zero_grad()
            preds = net(features.to(device))  # [bs,1,17,3]
            loss = loss_function(preds.to(device), labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # result = subprocess.run(
            #     ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            #     stdout=subprocess.PIPE, text=True, check=True)
            # memory_used, memory_total, temperature = map(int, result.stdout.strip().split(','))
            train_bar.set_description(f"train epoch [{epoch + 1}/{epochs}]")
            # train_bar.set_postfix_str(f"GPU Memory Used: {memory_used} MB / "
            #                           f"Memory Total: {memory_total} MB / "
            #                           f"Temperature: {temperature} °C")
        train_end_dt = datetime.datetime.now()
        train_cost = train_end_dt - train_start_dt
        train_cost_s = train_cost.seconds
        train_cost_mrs = train_cost.microseconds
        print(f"train epoch \033[1;31;40m[{epoch + 1}/{epochs}]\033[0m time_used "
              f"\033[1;31;40m{train_cost_s // (60 * 60)}h "
              f"{(train_cost_s % (60 * 60)) // 60}min "
              f"{(train_cost_s % (60 * 60)) % 60}s "
              f"{train_cost_mrs // 1000}ms\033[0m")

        # validate
        val_start_dt = datetime.datetime.now()
        net.eval()
        acc_mpjpe = 0.0
        acc_pampjpe = 0.0
        val_num = len(val_loader)
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout, leave=False)
            for batch_idx, batch_data in enumerate(val_bar):
                val_features = batch_data['input_wifi-csi']
                val_labels = batch_data['output']
                predict_y = net(val_features.to(device))
                cpu_predict_y = predict_y.cpu()
                mpjpe, pampjpe = calulate_error(cpu_predict_y.numpy(), val_labels.numpy())
                acc_mpjpe += mpjpe
                acc_pampjpe += pampjpe

                # result = subprocess.run(
                #     ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
                #      "--format=csv,noheader,nounits"],
                #     stdout=subprocess.PIPE, text=True, check=True)
                # memory_used, memory_total, temperature = map(int, result.stdout.strip().split(','))

                val_bar.set_description(f"val epoch [{epoch + 1}/{epochs}]")
                # val_bar.set_postfix_str(f"MPJPE = {mpjpe * 1000:.3f}, PA_MPJPE = {pampjpe * 1000:.3f}"
                #                         f"GPU Memory Used: {memory_used} MB / "
                #                         f"Memory Total: {memory_total} MB / "
                #                         f"Temperature: {temperature} °C")
        val_end_dt = datetime.datetime.now()
        val_cost = val_end_dt - val_start_dt
        val_cost_s = val_cost.seconds
        val_cost_mrs = val_cost.microseconds
        print(f"val epoch \033[1;31;40m[{epoch + 1}/{epochs}]\033[0m, time_used "
              f"\033[1;31;40m {val_cost_s // (60 * 60)}h "
              f"{(val_cost_s % (60 * 60)) // 60}min "
              f"{(val_cost_s % (60 * 60)) % 60}s "
              f"{val_cost_mrs // 1000}ms \033[0m")

        val_mpjpe = acc_mpjpe / val_num
        val_pampjpe = acc_pampjpe / val_num
        print('\033[1;31;40m [epoch %d / %d] \033[0m train_loss:\033[1;31;40m %.6f\033[0m val_pmpje:\033['
              '1;31;40m%.3f\033[0m val_pampjpe:\033[1;31;40m%.3f\033[0m' %
              (epoch + 1, epochs, running_loss / train_num, val_mpjpe * 1000, val_pampjpe * 1000))

        weight_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + f'_{epoch + 1}' + '.pth'
        torch.save(net.state_dict(), os.path.join('weight', weight_name))

        tb_writer.add_scalar('train_loss', running_loss / train_num, epoch + 1)
        tb_writer.add_scalar('MPJPE', val_mpjpe, epoch + 1)
        tb_writer.add_scalar('PA_MPJPE', val_pampjpe, epoch + 1)
        tb_writer.add_scalar('EPOCH_TRAIN_TIME', train_cost_s, epoch + 1)
        tb_writer.add_scalar('EPOCH_EVAL_TIME', val_cost_s, epoch + 1)
    pro_end_dt = datetime.datetime.now()
    pro_cost = pro_end_dt - pro_start_dt
    pro_cost_s = pro_cost.seconds
    pro_cost_mrs = pro_cost.microseconds
    print(
        f'Finished Training! time cost '
        f'{pro_cost_s // (60 * 60)}h '
        f'{(pro_cost_s % (60 * 60)) // 60}min '
        f'{(pro_cost_s % (60 * 60)) % 60}s '
        f'{pro_cost_mrs // 1000}ms')

    tb_writer.close()
