
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
import math


class SimpleTransformerLayer(nn.Module):
    """简化版Transformer层（1层）"""

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(SimpleTransformerLayer, self).__init__()
        self.out_dim = out_dim
        self.dropout = dropout

        # 自注意力层
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        # 输出层
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # 自注意力
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.out_dim)
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # 注意力输出
        out = torch.matmul(attn, v)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.out_proj(out)

        return out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        # ========== GCN分支（只使用1层） ==========
        # 疾病GCN
        self.gcn_dis = GCNConv(args.dis_in_channels, args.dis_hidden)

        # 代谢物GCN
        self.gcn_met = GCNConv(args.met_in_channels, args.met_hidden)

        # ========== Transformer分支（只使用1层） ==========
        # 疾病Transformer
        self.trans_dis = SimpleTransformerLayer(args.dis_in_channels, args.dis_hidden, dropout=args.dropout)

        # 代谢物Transformer
        self.trans_met = SimpleTransformerLayer(args.met_in_channels, args.met_hidden, dropout=args.dropout)

        # ========== 门控融合 ==========
        # 疾病门控融合
        self.gate_dis = nn.Sequential(
            nn.Linear(args.dis_hidden * 2, args.dis_hidden),
            nn.Sigmoid()
        )

        # 代谢物门控融合
        self.gate_met = nn.Sequential(
            nn.Linear(args.met_hidden * 2, args.met_hidden),
            nn.Sigmoid()
        )

        # ========== GATv2（只使用1层） ==========
        # 疾病GATv2
        self.gat_dis = GATv2Conv(args.dis_hidden, args.dis_out_channels, heads=1, concat=False,
                                 edge_dim=1, dropout=args.dropout)

        # 代谢物GATv2
        self.gat_met = GATv2Conv(args.met_hidden, args.met_out_channels, heads=1, concat=False,
                                 edge_dim=1, dropout=args.dropout)

        # ========== 对比学习投影层 ==========
        self.proj_dis = nn.Linear(args.dis_hidden, args.dis_hidden // 2)  # 降维以节省计算
        self.proj_met = nn.Linear(args.met_hidden, args.met_hidden // 2)

        # ========== 对比学习参数 ==========
        self.temperature = 0.5  # InfoNCE温度参数
        self.cont_weight = 0.1  # 对比学习权重

        self.dropout = args.dropout

    def contrastive_loss(self, z1, z2):
        """对比学习损失 (InfoNCE)"""
        batch_size = z1.size(0)

        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 相似度矩阵
        sim_matrix = torch.mm(z1, z2.T) / self.temperature

        # 正样本对（对角线）
        labels = torch.arange(batch_size).to(z1.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def forward(self, dis_data, met_data, pos_edge_index=None, neg_edge_index=None,
                compute_contrastive=True, return_output=True):
        """
        前向传播
        Args:
            compute_contrastive: 是否计算对比学习损失
            return_output: 是否返回输出矩阵
        Returns:
            如果return_output=True: (output, total_loss)
            如果return_output=False: total_loss
        """
        # ========== 疾病网络 ==========
        # GCN分支（1层）
        dis_gcn = F.dropout(dis_data.x, self.dropout, training=self.training)
        dis_gcn = self.gcn_dis(dis_gcn, dis_data.edge_index, dis_data.edge_attr)
        dis_gcn = dis_gcn.relu()

        # Transformer分支（1层）
        dis_trans = F.dropout(dis_data.x, self.dropout, training=self.training)
        dis_trans = self.trans_dis(dis_trans)
        dis_trans = dis_trans.relu()

        # ========== 代谢物网络 ==========
        # GCN分支（1层）
        met_gcn = F.dropout(met_data.x, self.dropout, training=self.training)
        met_gcn = self.gcn_met(met_gcn, met_data.edge_index, met_data.edge_attr)
        met_gcn = met_gcn.relu()

        # Transformer分支（1层）
        met_trans = F.dropout(met_data.x, self.dropout, training=self.training)
        met_trans = self.trans_met(met_trans)
        met_trans = met_trans.relu()

        # ========== 计算对比学习损失 ==========
        contrastive_loss = torch.tensor(0.0).to(dis_gcn.device)
        if compute_contrastive and self.training:
            # 投影特征用于对比学习
            dis_gcn_proj = self.proj_dis(dis_gcn)
            dis_trans_proj = self.proj_dis(dis_trans)
            met_gcn_proj = self.proj_met(met_gcn)
            met_trans_proj = self.proj_met(met_trans)

            # 计算对比损失（疾病网络 + 代谢物网络）
            cont_loss_dis = self.contrastive_loss(dis_gcn_proj, dis_trans_proj)
            cont_loss_met = self.contrastive_loss(met_gcn_proj, met_trans_proj)
            contrastive_loss = (cont_loss_dis + cont_loss_met) / 2

        # ========== 门控融合 ==========
        # 疾病门控融合
        dis_concat = torch.cat([dis_gcn, dis_trans], dim=-1)
        dis_gate = self.gate_dis(dis_concat)
        dis_fused = dis_gate * dis_gcn + (1 - dis_gate) * dis_trans

        # 代谢物门控融合
        met_concat = torch.cat([met_gcn, met_trans], dim=-1)
        met_gate = self.gate_met(met_concat)
        met_fused = met_gate * met_gcn + (1 - met_gate) * met_trans

        # ========== GATv2（1层） ==========
        dis_gat_out = self.gat_dis(dis_fused, dis_data.edge_index, dis_data.edge_attr)
        met_gat_out = self.gat_met(met_fused, met_data.edge_index, met_data.edge_attr)

        # ========== 输出 ==========
        output = torch.mm(met_gat_out, dis_gat_out.T)

        if return_output:
            return output, contrastive_loss
        else:
            return contrastive_loss

    # def decode(self, output, pos_edge_index, neg_edge_index):
    #     edge_index = torch.cat([pos_edge_index, neg_edge_index], 1)
    #
    #     # 更简洁的写法
    #     link_logits = output[edge_index[0], edge_index[1]]
    #
    #     return link_logits

    def decode(self, output, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], 1)
        link_logits = output[edge_index[0][0]][edge_index[1][0]]
        link_logits = link_logits.reshape(1, 1)
        for i in range(1, edge_index.shape[1]):
            link_logits = torch.cat((link_logits, output[edge_index[0][i]][edge_index[1][i]].reshape(1, 1)), 1)
        link_logits = link_logits.reshape(edge_index.shape[1])

        return link_logits


















# import torch
# from torch import nn
# from torch_geometric.nn import GCNConv, GATv2Conv
# import torch.nn.functional as F
#
#
# class Model(nn.Module):
#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.args = args
#         self.gcn_dis1 = GCNConv(self.args.dis_in_channels, self.args.dis_hidden)
#         self.gcn_dis2 = GCNConv(self.args.dis_hidden, self.args.dis_hidden)
#         self.gat_dis1 = GATv2Conv(self.args.dis_hidden, self.args.dis_out_channels, heads=1, concat=False, edge_dim=1,
#                                   dropout=args.dropout)
#
#         self.gcn_met1 = GCNConv(self.args.met_in_channels, self.args.met_hidden)
#         self.gcn_met2 = GCNConv(self.args.met_hidden, self.args.met_hidden)
#         self.gat_met1 = GATv2Conv(self.args.met_hidden, self.args.met_out_channels, heads=1, concat=False, edge_dim=1,
#                                 dropout=args.dropout)
#
#         self.dropout = args.dropout
#
#     def encode(self, dis_data, met_data):
#         # *************************************************疾病网络******************************************************
#         # 疾病GCN网络第一层
#         dis_x1 = F.dropout(dis_data.x, self.dropout, training=self.training)
#         dis_x1 = self.gcn_dis1(dis_x1, dis_data.edge_index, dis_data.edge_attr)
#         dis_x1 = dis_x1.relu()
#
#         # 疾病GCN网络第二层
#         dis_x2 = F.dropout(dis_x1, self.dropout, training=self.training)
#         dis_x2 = self.gcn_dis2(dis_x2, dis_data.edge_index, dis_data.edge_attr)
#         dis_x2 = dis_x2.relu()
#
#         # 疾病GAT网络层
#         dis_x3 = self.gat_dis1(dis_x1+dis_x2, dis_data.edge_index, dis_data.edge_attr)
#
#         # ************************************************代谢物网络*****************************************************
#         # 代谢物GCN网络第一层
#         met_x1 = F.dropout(met_data.x, self.dropout, training=self.training)
#         met_x1 = self.gcn_met1(met_x1, met_data.edge_index, met_data.edge_attr)
#         met_x1 = met_x1.relu()
#
#         # 代谢物GCN网络第二层
#         met_x2 = F.dropout(met_x1, self.dropout, training=self.training)
#         met_x2 = self.gcn_met2(met_x2, met_data.edge_index, met_data.edge_attr)
#         met_x2 = met_x2.relu()
#
#         # 代谢物GAT网络层
#         met_x3 = self.gat_met1(met_x1+met_x2, met_data.edge_index, met_data.edge_attr)
#
#         # **************************************************************************************************************
#         output = torch.mm(met_x3, dis_x3.T)
#
#         return output
#
#     def decode(self, output, pos_edge_index, neg_edge_index):
#         edge_index = torch.cat([pos_edge_index, neg_edge_index], 1)
#         link_logits = output[edge_index[0][0]][edge_index[1][0]]
#         link_logits = link_logits.reshape(1, 1)
#         for i in range(1, edge_index.shape[1]):
#             link_logits = torch.cat((link_logits, output[edge_index[0][i]][edge_index[1][i]].reshape(1, 1)), 1)
#         link_logits = link_logits.reshape(edge_index.shape[1])
#
#         return link_logits
