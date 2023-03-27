import torch
import numpy as np
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def data_loader(dataset, network):
    ppi_file = '../input/' + dataset + '/' + network + '/net.txt'
    protein_file = '../input/' + dataset + '/protein.txt'
    print('Reading data···\n')
    temp_ppi1 = []
    temp_ppi2 = []
    temp_ppi3 = []
    with open(ppi_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').split(',')
            temp_ppi1.append(line[0])
            temp_ppi2.append(line[1])
            temp_ppi3.append(line[2])

    with open(protein_file, 'r') as f:
        protein_list = []
        for line in f.readlines():
            protein_list.append(line.strip('\n'))
    protein_list = sorted(protein_list)
    protein_dict = {protein_list[i]: i for i in range(len(protein_list))}
    # 蛋白质总数量
    n = len(protein_list)

    pos_ppi = np.zeros((n, n))
    neg_ppi = np.zeros((n, n))

    for k in range(len(temp_ppi1)):
        i_index = protein_dict[temp_ppi1[k]]
        j_index = protein_dict[temp_ppi2[k]]
        if temp_ppi3[k] == '1':
            pos_ppi[i_index, j_index] = 1
            pos_ppi[j_index, i_index] = 1
        else:
            neg_ppi[i_index, j_index] = 1
            neg_ppi[j_index, i_index] = 1

    return [pos_ppi, neg_ppi], protein_list


def save_complex(f_star, protein, filepath):
    pred = f_star.cpu().numpy()
    save(pred, protein, filepath)


def save(pred, protein, path):
    with open(path, 'w') as f:
        for i in range(pred.shape[0]):
            ind = np.argwhere(pred[i] == 1).squeeze()
            for j in range(len(ind) - 1):
                f.write(protein[ind[j]] + ' ')
            f.write(protein[ind[-1]] + '\n')
