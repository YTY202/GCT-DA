import numpy as np
import random
import torch
from utils import get_link_labels

def train(model, optimizer, criterion, dis_data, met_data, train_pos_edge_index, train_matrix, cont_weight=0.1):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 训练集负采样，采集与正样本相同数量的负样本. 此部分应在train模块中，因为每次训练取不同的负样本可以提高训练效果。
    train_neg_edge_index = np.mat(np.where(train_matrix < 1)).T.tolist()
    random.shuffle(train_neg_edge_index)
    train_neg_edge_index = train_neg_edge_index[:train_pos_edge_index.shape[1]]
    train_neg_edge_index = np.array(train_neg_edge_index).T
    train_neg_edge_index = torch.tensor(train_neg_edge_index, dtype=torch.long).to(device)  # tensor格式训练集负样本

    optimizer.zero_grad()
    # output = model.encode(dis_data, met_data)
    output, contrastive_loss = model(dis_data, met_data, train_pos_edge_index, train_neg_edge_index)
    link_logits = model.decode(output, train_pos_edge_index, train_neg_edge_index)
    link_labels = get_link_labels(train_pos_edge_index, train_neg_edge_index).to(device)  # 训练集中正样本标签
    # loss = criterion(link_logits, link_labels)
    # loss.backward()
    # 主损失（链接预测）
    main_loss = criterion(link_logits, link_labels)

    # 总损失 = 主损失 + 对比学习损失（权重在模型中已设置）
    loss = main_loss + cont_weight * contrastive_loss

    loss.backward()
    optimizer.step()

    return loss, output