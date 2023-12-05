import dgl
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import pickle
import numpy as np
from tqdm import tqdm
# import random


class my_dataset(Dataset):
    def __init__(self, positive_pairs, negative_pairs):
        super(my_dataset, self).__init__()
        positive_pairs = torch.tensor(positive_pairs, dtype=torch.int64)
        positive_pairs = positive_pairs.permute(0, 2, 1)
        positive_pairs = positive_pairs.reshape(-1, positive_pairs.shape[-1])

        negative_pairs = torch.tensor(negative_pairs, dtype=torch.int64)
        negative_pairs = negative_pairs.permute(0, 2, 1)
        negative_pairs = negative_pairs.reshape(-1, negative_pairs.shape[-1])

        self.positive_pairs = positive_pairs[torch.randperm(
            positive_pairs.shape[0])]
        self.negative_pairs = negative_pairs[torch.randperm(
            negative_pairs.shape[0])]

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, index):
        return self.positive_pairs[index], self.negative_pairs[index]


class spatial_CL(nn.Module):
    def __init__(self, num_nodes, embed_size=128, embedding_pretrained=None):
        super(spatial_CL, self).__init__()
        if embedding_pretrained is not None:
            self.embeddings = nn.Embedding.from_pretrained(
                embedding_pretrained, freeze=False)
        else:
            self.embeddings = nn.Embedding(num_nodes, embed_size)

    def dist_func(self, o, d):
        return F.cosine_similarity(o, d, dim=0)

    def forward(self, pos_pair, neg_pair):
        # B * embed_size
        pos_node = self.embeddings(pos_pair[:, 0])
        pos_neigh = self.embeddings(pos_pair[:, 1])

        neg_node = self.embeddings(neg_pair[:, 0])
        neg_neigh = self.embeddings(neg_pair[:, 1])

        pos_dist = self.dist_func(pos_node, pos_neigh)
        neg_dist = self.dist_func(neg_node, neg_neigh)
        return pos_dist, neg_dist


# 制作对比学习样本
def get_spatial_contrastive_sample(graph, num_positive, num_negatives):
    positive_pairs = []
    negative_pairs = []
    N = graph.number_of_nodes()
    all_node_id = set(np.arange(N).tolist())

    for node in tqdm(range(N)):
        neighbors = graph.successors(node).numpy().tolist()
        # positive_size = min(num_positive, len(neighbors))
        positive_neigh = np.random.choice(
            neighbors,
            size=num_positive,
            replace=len(neighbors) < num_positive)
        positive_pairs.append([[node] * num_positive, positive_neigh.tolist()])

        non_neighbors = list(all_node_id - set(neighbors))
        negative_neigh = np.random.choice(
            non_neighbors,
            size=num_negatives,
            replace=len(non_neighbors) < num_negatives)
        negative_pairs.append([[node] * num_negatives,
                               negative_neigh.tolist()])

    return positive_pairs, negative_pairs


def main(embed_table_name='', embed_table_epoch=0):
    ds = '7day'
    task_name = 'simCL'
    print('task name: ', task_name)

    graph, _ = dgl.load_graphs(ws + f'graph_county_{ds}.pt')
    graph = graph[0]
    positive_samples, negative_samples = get_spatial_contrastive_sample(
        graph, 50, 200)

    dataset = my_dataset(positive_samples, negative_samples)
    dataloader = DataLoader(dataset=dataset, batch_size=512, drop_last=True)

    num_epoch = 80
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if embed_table_name:
        embed_table = torch.load(
            ws + f'embeds_{ds}_{embed_table_name}_e{embed_table_epoch}.pt',
            map_location=torch.device('cpu'))
        model = spatial_CL(num_nodes=graph.number_of_nodes(),
                           embed_size=embed_table.shape[1],
                           embedding_pretrained=embed_table)
    else:
        model = spatial_CL(num_nodes=graph.number_of_nodes(), embed_size=128)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('training...')
    epoch = 0
    for epoch in tqdm(range(num_epoch)):
        for batch in dataloader:
            optimizer.zero_grad()
            pos_pairs, neg_pairs = batch
            pos_pairs = pos_pairs.to(device)
            neg_pairs = neg_pairs.to(device)
            pos_dist, neg_dist = model(pos_pairs, neg_pairs)
            loss = (pos_dist - neg_dist).sum()
            loss.backward()
            optimizer.step()

    embed_tabel = model.embeddings.weight.data
    state_dict = model.state_dict()
    state_dict['embeddings.weight'] = embed_tabel

    if embed_table_name == '':
        table_name = f'embeds_{ds}_{task_name}_e{epoch+1}.pt'
    else:
        table_name = f'embeds_{ds}_{embed_table_name}_{task_name}_e{epoch + 1}.pt'
    torch.save(state_dict, 'D:/graphpretrainsimilarity/' + table_name)
    print('embedding table saved')
    print('pretrain task name: ', task_name)
    print('use pretrained embedding: ',
          'None' if embed_table_name == '' else embed_table_name)


if __name__ == '__main__':
    ws = 'D:/深度学习/大创/数据预处理与可视化/'
    main('simCL', 80)  # linkPre 82
