import numpy as np
import pandas as pd
import random
import time
from param import *
from utils import *
from model import *
from train import *
from test import *
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

Adj = pd.read_excel('../data/association_matrix.xlsx', header=0)
print(f'疾病代谢物邻接矩阵：\n{Adj}')

Dis_simi = pd.read_excel('../data/diease_network.xlsx', header=0)
print(f'疾病相似性矩阵：\n{Dis_simi}')

Meta_simi = pd.read_excel('../data/metabolite_network.xlsx', header=0)
print(f'代谢物相似性矩阵：\n{Meta_simi}')

args = parameter_parser()

index_matrix = np.mat(np.where(Adj == 1))  # 输出邻接矩阵中为“1”的关联关系，维度：2 X 4763
association_nam = index_matrix.shape[1]  # 关联关系数：4763
random_index = index_matrix.T.tolist()  # list：4763 X 2
random.seed(args.seed)  # random.seed(): 设定随机种子，使得random.shuffle随机打乱的顺序一致
random.shuffle(random_index)  # random.shuffle将random_index列表中的元素打乱顺序
k_folds = 5
CV_size = int(association_nam / k_folds)  # 每折的个数
temp = np.array(random_index[:association_nam - association_nam %
                              k_folds]).reshape(k_folds, CV_size, -1).tolist()  # %取余
temp[k_folds - 1] = temp[k_folds - 1] + \
                    random_index[association_nam - association_nam % k_folds:]  # 将余下的元素加到最后一折里面
random_index = temp
metric = np.zeros((1, 7))

best_metric = [{}, {}, {}, {}, {}]

print(f"seed={args.seed}, dropout={args.dropout} evaluating met-disease....")
for k in range(k_folds):
    print("------this is %dth cross validation------" % (k + 1))
    train_matrix = np.matrix(Adj, copy=True)  # 将邻接矩阵转化为np矩阵
    dis_matrix = np.matrix(Dis_simi, copy=True)  # 将疾病相似性矩阵转化为np矩阵
    met_matrix = np.matrix(Meta_simi, copy=True)  # 将代谢物相似性矩阵转化为np矩阵

    val_pos_edge_index = np.array(random_index[k]).T  # 验证集边索引，正样本
    val_pos_edge_index = torch.tensor(val_pos_edge_index, dtype=torch.long)  # tensor格式，验证集正样本
    # 验证集负采样，采集与正样本相同数量的负样本
    val_neg_edge_index = np.mat(np.where(train_matrix < 1)).T.tolist()
    random.shuffle(val_neg_edge_index)
    val_neg_edge_index = val_neg_edge_index[:val_pos_edge_index.shape[1]]
    val_neg_edge_index = np.array(val_neg_edge_index).T
    val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long)  # tensor格式，验证集负样本

    train_matrix[tuple(np.array(random_index[k]).T)] = 0  # tuple()转化为元组，将train_matrix中每一折中的测试集元素变为0
    train_pos_edge_index = np.mat(np.where(train_matrix > 0))  # 训练集边索引，正样本
    train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long)  # tensor格式，训练集正样本

    # --------------------------------------------疾病Data数据构建--------------------------------------------------
    dis_x_list = (train_matrix.T).tolist()
    dis_edge_index_list = np.mat(np.where(dis_matrix > 0)).tolist()  # 疾病边索引的List格式
    dis_matrix_list = dis_matrix.tolist()  # 疾病网络的list格式，用来寻找边权值
    dis_edge_attr_list = []  # 代谢物边权值List空表，用来存放边权值
    for i in range(len(dis_edge_index_list[0])):  # for循环用来寻找疾病网络中的边权值
        row = dis_edge_index_list[0][i]
        col = dis_edge_index_list[1][i]
        dis_edge_attr_list.append(dis_matrix_list[row][col])
    dis_edge_attr = torch.tensor(dis_edge_attr_list, dtype=torch.float)
    dis_data = getData(dis_x_list, dis_edge_index_list, dis_edge_attr)  # 疾病data格式数据构建完成
    print(f'dis_data: {dis_data}')

    # -------------------------------------------代谢物Data数据构建--------------------------------------------------
    met_x_list = train_matrix.tolist()
    met_edge_index_list = np.mat(np.where(met_matrix > 0)).tolist()  # 代谢物边索引的List格式
    met_matrix_list = met_matrix.tolist()  # 代谢物网络的list格式，用来寻找边权值
    met_edge_attr_list = []  # 代谢物边权值List空表，用来存放边权值
    for i in range(len(met_edge_index_list[0])):  # for循环用来寻找代谢物网络中的边权值
        row = met_edge_index_list[0][i]
        col = met_edge_index_list[1][i]
        met_edge_attr_list.append(met_matrix_list[row][col])
    met_edge_attr = torch.tensor(met_edge_attr_list, dtype=torch.float)
    met_data = getData(met_x_list, met_edge_index_list, met_edge_attr)  # 代谢物data格式数据构建完成
    print(f'met_data: {met_data}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = Model(args=args).to(device)
    dis_data = dis_data.to(device)
    met_data = met_data.to(device)
    train_pos_edge_index = train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = F.binary_cross_entropy_with_logits

    best_auc = best_prc = best_acc = 0

    for epoch in range(0, args.epochs):
        start = time.time()
        if epoch == 0:
            best_auc = 0
        train_loss, output = train(model, optimizer, criterion, dis_data, met_data, train_pos_edge_index, train_matrix)
        auc, acc, prc, pre, rec, f1, tpr, fpr, recall, precision = mytest(model, val_pos_edge_index, val_neg_edge_index, output)
        if auc >= best_auc and prc >= best_prc and acc >= best_acc:
            best_auc = auc
            best_prc = prc
            best_acc = acc
            tpr = tpr.tolist()
            fpr = fpr.tolist()
            precision = precision.tolist()
            recall = recall.tolist()
            best_metric[k] = {'AUC': auc, 'ACC': acc, 'PRC': prc, 'PRE': pre, 'REC': rec, 'F1': f1, 'tpr': tpr, 'fpr': fpr, 'recall': recall, 'precision': precision}
        end = time.time()
        print(f'Epoch:{epoch+1}  loss:{train_loss:.4f}  AUC:{auc:.4f}  ACC:{acc:.4f}  PRC:{prc:.4f}  PRE:{pre:.4f}  REC:{rec:.4f}  F1:{f1:.4f}  time: {(end - start):.2f}')
    # print(best_metric[k])

print('*****************************************************************************************************************')
print('*****************************************************************************************************************')
print(f'best_metric: {best_metric}')

AUC = ACC = PRC = PRE = REC = F1 = 0
for i in range(k_folds):
    AUC += best_metric[i]['AUC']
    ACC += best_metric[i]['ACC']
    PRC += best_metric[i]['PRC']
    PRE += best_metric[i]['PRE']
    REC += best_metric[i]['REC']
    F1 += best_metric[i]['F1']
print('*******************************************************************************************************************')
print('*******************************************************************************************************************')
average_5 = {'AUC': AUC/5, 'ACC': ACC/5, 'PRC': PRC/5, 'PRE': PRE/5, 'REC': REC/5, 'F1': F1/5}
print(f'{k_folds}_average_metric: {average_5}')
print('-------------------------------------- End of Code --------------------------------------')