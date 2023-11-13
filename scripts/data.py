import datetime
import os
import argparse
import sys

import yaml
import numpy as np
import torch

from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error

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
    tb_writer = SummaryWriter(log_dir=log_dir_name)
    dataset_root = args.dataset_root
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    train_dataset, val_dataset = make_dataset(dataset_root, config)

    rng_generator = torch.manual_seed(config['init_rand_seed'])
    train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['train_loader'])
    val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['validation_loader'])

    # TODO: Settings, e.g., your model, optimizer, device, ...
    # [17,3] [17, 3]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f'using {device} device!')

    net = resnet34()
    net.to(device)
    # model_weight_path = './resnet34.pth'
    # if not os.path.exists(model_weight_path):
    #     print("file {} does not exist.".format(model_weight_path))
    #     model_weight_path = '../03_ResNet/resnet34-pre.pth'
    #     print("using file {} pre-train.".format(model_weight_path))
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    loss_function = torch.nn.MSELoss()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    # TODO: Codes for training (and saving models)
    # Just an example for illustration.
    epochs = config['epochs']
    best_acc = 0.0
    save_path = './resnet34.pth'
    for epoch in range(epochs):
        # Please check the data structure here.
        # train
        net.train()
        running_loss = 0.0
        # train_bar = tqdm(train_loader, file=sys.stdout)
        train_num = len(train_loader)
        for batch_idx, batch_data in enumerate(train_loader):
            features = batch_data['input_wifi-csi']
            labels = batch_data['output']
            optimizer.zero_grad()
            preds = net(features.to(device))
            loss = loss_function(preds.to(device), labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print("train epoch [{}/{}] loss:{:.3f}".format(epoch + 1,
                                                           epochs,
                                                           loss))
        # validate
        net.eval()
        acc_mpjpe = 0.0
        val_num = len(val_loader)
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for batch_idx, batch_data in enumerate(val_loader):
                val_features = batch_data['input_wifi-csi']
                val_labels = batch_data['output']
                predict_y = net(val_features.to(device))
                mpjpe = calulate_error(predict_y, val_labels.to(device))
                acc_mpjpe += mpjpe

        val_mpjpe = acc_mpjpe / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_num, val_mpjpe))

        tb_writer.add_scalar('train_loss', running_loss / train_num, epoch + 1)
        tb_writer.add_scalar('MPJPE', mpjpe, epoch + 1)

    print('Finished Training')
