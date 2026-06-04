import os
import glob
import random
import numpy as np
import pandas as pd
import torch

from param import parameter_parser
from utils import getData
from model import Model
from test import mytest

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# =========================
# 路径设置
# =========================
TRAIN_ADJ_PATH = '../data/association_matrix.xlsx'   # 训练时使用的关联矩阵，用来提供“训练空间”的实体顺序
EXTERNAL_ADJ_PATH = '../validate_data/dis_meta_association_matrix.xlsx'
EXTERNAL_DIS_SIM_PATH = '../validate_data/disease_com_similarity.xlsx'
EXTERNAL_MET_SIM_PATH = '../validate_data/metabolite_com_similarity.xlsx'

MODEL_DIR = './best_model'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 0
TEST_RATIO = 0.2
NEG_POS_RATIO = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_named_matrix(path):
    """
    读取带行名/列名的 Excel 矩阵
    """
    df = pd.read_excel(path, header=0, index_col=0)

    # 清理空行空列
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # 统一名称格式
    df.index = df.index.map(lambda x: str(x).strip())
    df.columns = df.columns.map(lambda x: str(x).strip())

    # 去掉可能残留的 unnamed
    df = df.loc[[i for i in df.index if not str(i).lower().startswith('unnamed')], :]
    df = df.loc[:, [c for c in df.columns if not str(c).lower().startswith('unnamed')]]

    # 全转数值，转不了变 NaN，再补 0
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df


def inspect_checkpoint_input_dims(model_path, device):
    """
    从保存的 checkpoint 里读取模型要求的输入维度
    返回:
        expected_num_mets, expected_num_dis
    含义:
        疾病分支输入维度 = 训练代谢物数
        代谢物分支输入维度 = 训练疾病数
    """
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    expected_num_mets = state_dict['gcn_dis.lin.weight'].shape[1]
    expected_num_dis = state_dict['gcn_met.lin.weight'].shape[1]

    return expected_num_mets, expected_num_dis


def project_external_to_train_space(train_adj_df, ext_adj_df, ext_dis_df, ext_met_df):
    """
    把外部数据映射到训练空间:
    - 输出 full_adj / full_dis / full_met 的 shape 与训练时一致
    - 只在重叠实体区域填入外部数据
    - 其余位置补 0，节点自身对角线补 1
    - 同时返回 valid_mask，后面只在“真实外部覆盖区域”采负样本
    """
    train_mets = [str(x).strip() for x in train_adj_df.index]
    train_dis = [str(x).strip() for x in train_adj_df.columns]

    ext_adj_df = ext_adj_df.copy()
    ext_dis_df = ext_dis_df.copy()
    ext_met_df = ext_met_df.copy()

    # 找到同时能在外部文件中对齐的实体
    common_mets = [
        m for m in train_mets
        if m in ext_adj_df.index and m in ext_met_df.index and m in ext_met_df.columns
    ]
    common_dis = [
        d for d in train_dis
        if d in ext_adj_df.columns and d in ext_dis_df.index and d in ext_dis_df.columns
    ]

    if len(common_mets) == 0:
        raise ValueError('训练集与外部数据没有可对齐的代谢物名称。')
    if len(common_dis) == 0:
        raise ValueError('训练集与外部数据没有可对齐的疾病名称。')

    n_met = len(train_mets)
    n_dis = len(train_dis)

    # 1) 关联矩阵：扩展到训练空间
    full_adj = pd.DataFrame(
        np.zeros((n_met, n_dis), dtype=np.float32),
        index=train_mets,
        columns=train_dis
    )
    full_adj.loc[common_mets, common_dis] = ext_adj_df.loc[common_mets, common_dis].values

    # 2) 疾病相似矩阵：扩展到训练空间
    full_dis = pd.DataFrame(
        np.zeros((n_dis, n_dis), dtype=np.float32),
        index=train_dis,
        columns=train_dis
    )
    np.fill_diagonal(full_dis.values, 1.0)
    full_dis.loc[common_dis, common_dis] = ext_dis_df.loc[common_dis, common_dis].values

    # 3) 代谢物相似矩阵：扩展到训练空间
    full_met = pd.DataFrame(
        np.zeros((n_met, n_met), dtype=np.float32),
        index=train_mets,
        columns=train_mets
    )
    np.fill_diagonal(full_met.values, 1.0)
    full_met.loc[common_mets, common_mets] = ext_met_df.loc[common_mets, common_mets].values

    # 4) 有效评估区域 mask：只在 overlap 子空间评估
    valid_mask = pd.DataFrame(
        np.zeros((n_met, n_dis), dtype=bool),
        index=train_mets,
        columns=train_dis
    )
    valid_mask.loc[common_mets, common_dis] = True

    return full_adj, full_dis, full_met, valid_mask, common_mets, common_dis


def split_external_dataset(full_adj, valid_mask, test_ratio=0.2, neg_pos_ratio=1.0, seed=0):
    """
    只在 valid_mask == True 的区域里:
    - 划分测试正样本
    - 采样测试负样本
    - support_adj 中把测试正样本抹掉
    """
    rng = np.random.RandomState(seed)

    valid_mask = valid_mask.astype(bool)

    pos_edges = np.argwhere((full_adj == 1) & valid_mask)
    neg_edges = np.argwhere((full_adj < 1) & valid_mask)

    rng.shuffle(pos_edges)
    rng.shuffle(neg_edges)

    if len(pos_edges) == 0:
        raise ValueError('外部数据在重叠实体区域中没有正样本，无法验证。')

    test_pos_num = max(1, int(len(pos_edges) * test_ratio))
    test_neg_num = int(test_pos_num * neg_pos_ratio)

    if len(neg_edges) < test_neg_num:
        raise ValueError(f'可采样负样本不足：需要 {test_neg_num}，实际只有 {len(neg_edges)}。')

    test_pos_edges = pos_edges[:test_pos_num]
    test_neg_edges = neg_edges[:test_neg_num]

    support_adj = np.array(full_adj, copy=True)
    support_adj[test_pos_edges[:, 0], test_pos_edges[:, 1]] = 0

    test_pos_edge_index = torch.tensor(test_pos_edges.T, dtype=torch.long)
    test_neg_edge_index = torch.tensor(test_neg_edges.T, dtype=torch.long)

    return support_adj, test_pos_edge_index, test_neg_edge_index


def build_graph_data(train_matrix_np, dis_sim_np, met_sim_np):
    """
    与训练阶段保持一致:
    dis_x_list = train_matrix.T
    met_x_list = train_matrix
    """
    train_matrix = np.matrix(train_matrix_np, copy=True)
    dis_matrix = np.matrix(dis_sim_np, copy=True)
    met_matrix = np.matrix(met_sim_np, copy=True)

    # 疾病图
    dis_x_list = train_matrix.T.tolist()
    dis_edge_index_list = np.mat(np.where(dis_matrix > 0)).tolist()
    dis_matrix_list = dis_matrix.tolist()
    dis_edge_attr_list = []
    for i in range(len(dis_edge_index_list[0])):
        row = dis_edge_index_list[0][i]
        col = dis_edge_index_list[1][i]
        dis_edge_attr_list.append(dis_matrix_list[row][col])

    dis_edge_attr = torch.tensor(dis_edge_attr_list, dtype=torch.float)
    dis_data = getData(dis_x_list, dis_edge_index_list, dis_edge_attr)

    # 代谢物图
    met_x_list = train_matrix.tolist()
    met_edge_index_list = np.mat(np.where(met_matrix > 0)).tolist()
    met_matrix_list = met_matrix.tolist()
    met_edge_attr_list = []
    for i in range(len(met_edge_index_list[0])):
        row = met_edge_index_list[0][i]
        col = met_edge_index_list[1][i]
        met_edge_attr_list.append(met_matrix_list[row][col])

    met_edge_attr = torch.tensor(met_edge_attr_list, dtype=torch.float)
    met_data = getData(met_x_list, met_edge_index_list, met_edge_attr)

    train_pos_edge_index = np.mat(np.where(train_matrix > 0))
    train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long)

    return train_matrix, dis_data, met_data, train_pos_edge_index


def load_model(model_path, device):
    args = parameter_parser()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'args' in checkpoint and isinstance(checkpoint['args'], dict):
        for k, v in checkpoint['args'].items():
            try:
                setattr(args, k, v)
            except Exception:
                pass

    model = Model(args=args).to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        meta = {
            'fold': checkpoint.get('fold', None),
            'epoch': checkpoint.get('epoch', None),
            'best_auc': checkpoint.get('best_auc', None),
        }
    else:
        model.load_state_dict(checkpoint)
        meta = {
            'fold': None,
            'epoch': None,
            'best_auc': None,
        }

    model.eval()
    return model, meta


def evaluate_one_model(model_path, support_adj, dis_sim, met_sim,
                       test_pos_edge_index, test_neg_edge_index, device):
    train_matrix, dis_data, met_data, train_pos_edge_index = build_graph_data(
        support_adj, dis_sim, met_sim
    )

    dis_data = dis_data.to(device)
    met_data = met_data.to(device)
    train_pos_edge_index = train_pos_edge_index.to(device)
    test_pos_edge_index = test_pos_edge_index.to(device)
    test_neg_edge_index = test_neg_edge_index.to(device)

    model, meta = load_model(model_path, device)

    with torch.no_grad():
        output, contrastive_loss = model(
            dis_data,
            met_data,
            pos_edge_index=None,
            neg_edge_index=None,
            compute_contrastive=False,
            return_output=True
        )

    auc, acc, prc, pre, rec, f1, tpr, fpr, recall, precision = mytest(
        model, test_pos_edge_index, test_neg_edge_index, output
    )

    result = {
        'model_path': model_path,
        'fold': meta['fold'],
        'epoch': meta['epoch'],
        'saved_best_auc': meta['best_auc'],
        'AUC': auc,
        'ACC': acc,
        'PRC': prc,
        'PRE': pre,
        'REC': rec,
        'F1': f1
    }
    return result


def main():
    set_seed(SEED)

    model_paths = sorted(glob.glob(os.path.join(MODEL_DIR, '*.pth')))
    if len(model_paths) == 0:
        raise FileNotFoundError(f'没有在 {MODEL_DIR} 下找到 .pth 模型文件')

    # 先从 checkpoint 读出模型真正要求的输入维度
    expected_num_mets, expected_num_dis = inspect_checkpoint_input_dims(model_paths[0], DEVICE)
    print(f'模型要求输入维度 -> 训练代谢物数: {expected_num_mets}, 训练疾病数: {expected_num_dis}')

    print('================ 读取训练集实体空间 ================')
    train_adj_df = read_named_matrix(TRAIN_ADJ_PATH)
    print(f'训练关联矩阵 shape: {train_adj_df.shape}')

    if train_adj_df.shape != (expected_num_mets, expected_num_dis):
        raise ValueError(
            f'训练关联矩阵读取后 shape={train_adj_df.shape}，'
            f'但 checkpoint 要求 shape=({expected_num_mets}, {expected_num_dis})。'
            f'请先检查 TRAIN_ADJ_PATH 是否正确，或该文件是否也需要调整读取方式。'
        )

    print('================ 读取外部数据集 ================')
    ext_adj_df = read_named_matrix(EXTERNAL_ADJ_PATH)
    ext_dis_df = read_named_matrix(EXTERNAL_DIS_SIM_PATH)
    ext_met_df = read_named_matrix(EXTERNAL_MET_SIM_PATH)

    print(f'原始外部关联矩阵 shape: {ext_adj_df.shape}')
    print(f'原始外部疾病相似矩阵 shape: {ext_dis_df.shape}')
    print(f'原始外部代谢物相似矩阵 shape: {ext_met_df.shape}')

    print('================ 投影到训练空间 ================')
    adj_df, dis_df, met_df, valid_mask_df, common_mets, common_dis = project_external_to_train_space(
        train_adj_df, ext_adj_df, ext_dis_df, ext_met_df
    )

    print(f'训练-外部重叠代谢物数: {len(common_mets)}')
    print(f'训练-外部重叠疾病数: {len(common_dis)}')
    print(f'投影后关联矩阵 shape: {adj_df.shape}')
    print(f'投影后疾病相似矩阵 shape: {dis_df.shape}')
    print(f'投影后代谢物相似矩阵 shape: {met_df.shape}')

    full_adj = adj_df.values.astype(np.float32)
    dis_sim = dis_df.values.astype(np.float32)
    met_sim = met_df.values.astype(np.float32)
    valid_mask = valid_mask_df.values.astype(bool)

    # 再确认一次，和模型期望维度一致
    if full_adj.shape != (expected_num_mets, expected_num_dis):
        raise ValueError(
            f'投影后 full_adj.shape={full_adj.shape}，'
            f'但模型要求 ({expected_num_mets}, {expected_num_dis})'
        )

    print('================ 划分外部测试集 ================')
    support_adj, test_pos_edge_index, test_neg_edge_index = split_external_dataset(
        full_adj,
        valid_mask,
        test_ratio=TEST_RATIO,
        neg_pos_ratio=NEG_POS_RATIO,
        seed=SEED
    )

    print(f'测试正样本数: {test_pos_edge_index.shape[1]}')
    print(f'测试负样本数: {test_neg_edge_index.shape[1]}')
    print(f'support 图中剩余正样本数: {int(np.sum(support_adj == 1))}')

    print('================ 加载模型并外部验证 ================')
    all_results = []
    for model_path in model_paths:
        print(f'\n>>>> 正在验证模型: {model_path}')
        result = evaluate_one_model(
            model_path=model_path,
            support_adj=support_adj,
            dis_sim=dis_sim,
            met_sim=met_sim,
            test_pos_edge_index=test_pos_edge_index,
            test_neg_edge_index=test_neg_edge_index,
            device=DEVICE
        )
        all_results.append(result)

        print(
            f"AUC={result['AUC']:.4f}  "
            f"ACC={result['ACC']:.4f}  "
            f"PRC={result['PRC']:.4f}  "
            f"PRE={result['PRE']:.4f}  "
            f"REC={result['REC']:.4f}  "
            f"F1={result['F1']:.4f}"
        )

    result_df = pd.DataFrame(all_results)
    print('\n================ 每个模型的外部验证结果 ================')
    print(result_df)

    mean_result = result_df[['AUC', 'ACC', 'PRC', 'PRE', 'REC', 'F1']].mean()
    print('\n================ 平均外部验证结果 ================')
    print(mean_result)

    result_df.to_excel('external_validation_results.xlsx', index=False)
    mean_result.to_frame(name='mean').to_excel('external_validation_mean.xlsx')

    print('\n结果已保存到当前目录：')
    print('external_validation_results.xlsx')
    print('external_validation_mean.xlsx')


if __name__ == '__main__':
    main()