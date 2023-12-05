import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RegressionDataset(Dataset):
    def __init__(self, data):
        self.features = data[:, :-1]
        self.labels = data[:, -1:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).int(), \
            torch.from_numpy(self.labels[idx]).float()


class dnn_predictor(nn.Module):
    def __init__(self, hidden_size, output_size, num_cp, num_seller):
        super(dnn_predictor, self).__init__()
        self.cp_embed = nn.Embedding(num_cp, 32)
        self.week_embed = nn.Embedding(7, 16)
        self.hour_embed = nn.Embedding(24, 16)
        self.seller_embed = nn.Embedding(num_seller, 32)

        self.mlp = nn.Sequential(nn.Linear(96 + 7, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size,
                                           hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, output_size))

    def forward(self, x):
        cp = self.cp_embed(x[:, 0])
        week = self.week_embed(x[:, 1])
        hour = self.hour_embed(x[:, 2])
        seller = self.seller_embed(x[:, 3])
        input_fea = torch.concat([cp, week, hour, seller, x[:, 4:]], dim=1)

        return self.mlp(input_fea)


def main():
    # df = pd.read_csv(ws + f'cleaned_data.csv')
    # data = df.values
    train_df = pd.read_csv(ws + 'train.csv')  # hf+
    train_data = train_df.values  # hf+

    val_df = pd.read_csv(ws + 'valid.csv')  # hf+
    val_data = val_df.values  # hf+

    test_df = pd.read_csv(ws + 'test.csv')  # hf+
    test_data = test_df.values  # hf+

    # data_size = len(data)
    # train_size = int(data_size * 0.7)
    # val_size = int(data_size * 0.9)

    # random.seed(1)
    # random.shuffle(data)
    train_dataset = RegressionDataset(train_data)  # hf改
    val_dataset = RegressionDataset(val_data)  # hf改
    test_dataset = RegressionDataset(test_data)  # hf改

    batch_size = 256
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    num_epoch = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    hidden_size = 128
    num_cp = max(int(train_data[:, 0].max()), int(val_data[:, 0].max()),
                 int(test_data[:, 0].max())) + 1  # hf+
    num_seller = max(int(train_data[:, 3].max()), int(val_data[:, 3].max()),
                     int(test_data[:, 3].max())) + 1  # hf+

    model = dnn_predictor(
        hidden_size=hidden_size,
        output_size=1,
        num_cp=num_cp,  # int(data[:, 0].max()) + 1
        num_seller=num_seller)  # int(data[:, 3].max()) + 1
    model.load_state_dict(torch.load(ws + 'embeds_7day_simCL_simCL_e80.pt',
                                     map_location=torch.device('cpu')),
                          strict=False)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    print('training...')
    for epoch in range(num_epoch):
        model.train()
        for data in tqdm(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            eta = model(inputs)
            loss = criterion(eta, labels)
            loss.backward()
            optimizer.step()

        rmse_lst = []
        mae_lst = []
        mape_lst = []
        model.eval()
        for data in tqdm(val_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)

            y = labels.cpu().numpy()
            y_hat = predictions.squeeze().cpu().detach().numpy()
            rmse_lst.append(mean_squared_error(y, y_hat, squared=False))
            mae_lst.append(mean_absolute_error(y, y_hat))
            mape_lst.append((100 * abs(y - y_hat)).mean())

        print(epoch + 1)
        print(f'rmse: {np.mean(rmse_lst)}    '
              f'mae: {np.mean(mae_lst)}    '
              f'mape: {np.mean(mape_lst)}\n')

    rmse_lst = []
    mae_lst = []
    mape_lst = []
    model.eval()
    for data in test_dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)

        y = labels.cpu().numpy()
        y_hat = predictions.squeeze().cpu().detach().numpy()
        rmse_lst.append(mean_squared_error(y, y_hat, squared=False))
        mae_lst.append(mean_absolute_error(y, y_hat))
        mape_lst.append((100 * abs(y - y_hat)).mean())

    print('test')
    print(f'rmse: {np.mean(rmse_lst)}    '
          f'mae: {np.mean(mae_lst)}    '
          f'mape: {np.mean(mape_lst)}\n')


if __name__ == '__main__':
    ws = 'D:/深度学习/大创/数据预处理与可视化/'
    main()
