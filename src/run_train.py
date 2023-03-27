import torch
import time
import os

from model import Model
from utils import setup_seed, data_loader, save_complex


# 参数设置
beta = 4
lamda = 32
gamma = 0.5
K = 200
eta = 0.5
tau = 0.1
seed = 115
ite = 200
rho = 1e-6
repeat_times = 1

dataset = 'dataset6'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据
network1, _ = data_loader(dataset, 'net1')
network2, protein = data_loader(dataset, 'net2')


setup_seed(seed)

print("Parameter setting:")
print("seed:%s K:%d beta:%s lamda:%s gamma:%s repeat_times:%d iter:%d tau:%.1f" % (
    str(seed), K, str(beta), str(lamda), str(gamma), repeat_times, ite, tau ))

# 实例化模型
model = Model(network1, network2, device, beta, lamda, gamma, K, eta, tau, ite, rho, repeat_times)

# 计时
t1 = time.time()

# 训练
F_hat, F_star, F_star_A, F_star_B, C_star, S_star_A, S_star_B = model.train()

t2 = time.time()
print("runtime: %f" % (t2 - t1))

tmp = torch.sum(F_star, dim=1)
tmp1 = torch.sum(F_star_A, dim=1)
tmp2 = torch.sum(F_star_B, dim=1)
print("complex num: %d  protein num: %d" % (F_star.shape[1], len(tmp[tmp > 0])))
print("Net1 C_complex num: %d  S_complex num: %d sum: %d protein num: %d" % (
    C_star.shape[1], S_star_A.shape[1], (F_star_A.shape[1]), len(tmp1[tmp1 > 0])))
print("Net2 C_complex num: %d  S_complex num: %d sum: %d protein num: %d" % (
    C_star.shape[1], S_star_B.shape[1], (F_star_B.shape[1]), len(tmp2[tmp2 > 0])))


save_path = './' + dataset
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_complex(F_star.t(), protein, save_path + '/complex.txt')
save_complex(F_star_A.t(), protein, save_path + '/complex1.txt')
save_complex(F_star_B.t(), protein, save_path + '/complex2.txt')
