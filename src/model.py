import torch

dtype = torch.float64


class Model:
    def __init__(self, network1, network2, device, beta=0.5, lamda=0.5, gamma=0.5, K=1000, eta=0.5, tau=0.3, iter=200, rho=1e-6, repeat_times=1):
        self.device = device

        self.ap = [torch.from_numpy(network1[0]).to(self.device),
                   torch.from_numpy(network2[0]).to(self.device)]
        self.an = [torch.from_numpy(network1[1]).to(self.device),
                   torch.from_numpy(network2[1]).to(self.device)]
        
        self.N = network1[0].shape[0]      # 返回行数,蛋白质数量

        self.beta = torch.tensor(beta, device=self.device, dtype=dtype)
        self.lamda = torch.tensor(lamda, device=self.device, dtype=dtype)
        self.gamma = torch.tensor(gamma * (self.N ** 2), device=self.device, dtype=dtype)

        self.rho = torch.tensor(rho, device=self.device, dtype=dtype)

        self.K = K
        self.tau = tau
        self.iter = iter
        self.repeat_times = repeat_times

        self.num_views = 2

        self.Kc = int(eta * self.K)
        self.Ks = int((1 / self.num_views) * (self.K-self.Kc))

        del network1
        del network2

        self.H = torch.eye(self.N, device=self.device, dtype=dtype) - torch.full((self.N, self.N), 1 / self.N,
                                                                                 device=self.device, dtype=dtype)

        self.eps = torch.tensor(1e-8, device=self.device, dtype=dtype)

        self.lowest_score = torch.tensor(float('inf'), device=self.device, dtype=dtype)

    def initial_variable(self):
        # print("initial variable...")
        self.C_old = torch.rand((self.N, self.Kc), device=self.device, dtype=dtype)
        self.S_old = []
        self.theta = []
        self.D = []
        self.part1 = []

        for i in range(self.num_views):
            self.D.append(torch.diag(torch.sum(torch.abs(self.ap[i] + self.an[i]), dim=0)))

            self.S_old.append(torch.rand((self.N, self.Ks), device=self.device, dtype=dtype))

            self.theta.append(torch.zeros((self.N, 1), device=self.device, dtype=dtype))
            self.theta[i][torch.sum(self.ap[i], dim=0) > 0] = 1
            self.theta[i] = self.theta[i].mm(self.theta[i].t())

            self.part1.append(self.theta[i] * self.ap[i])

        self.S = [self.S_old[0].clone(), self.S_old[1].clone()]

    def train(self):

        for i in range(self.repeat_times):
            # print('This is the ', str(i), '-th repeat...')
            self.initial_variable()

            score = torch.tensor(0, device=self.device, dtype=dtype)
            F_old = torch.cat((self.C_old, self.S_old[0], self.S_old[1]), 1)

            # print('training...')
            for j in range(self.iter):
                Wc = 0
                Tc = 0
                Dc = self.theta[0].mm(self.C_old) + self.theta[1].mm(self.C_old)

                for m in range(self.num_views):
                    tmp = torch.cat((self.C_old, self.S_old[m]), 1)
                    part2 = tmp.mm(tmp.t()) + self.eps
                    part1_dot_divide_part2 = self.part1[m] / part2

                    del tmp
                    del part2

                    Wc += part1_dot_divide_part2.mm(self.C_old)
                    Wc += self.beta * (self.ap[m].mm(self.C_old))

                    Tc += self.beta * (self.D[m].mm(self.C_old) + self.an[m].mm(self.C_old))

                    Ws = part1_dot_divide_part2.mm(self.S_old[m])
                    Ws += self.beta * (self.ap[m].mm(self.S_old[m]))

                    part3 = self.beta * (self.D[m].mm(self.S_old[m]) + self.an[m].mm(self.S_old[m]))

                    if m == 0:
                        part4 = self.gamma * ((self.N - 1) ** (-2)) * self.H.mm(self.S_old[m + 1]).mm(
                                self.S_old[m + 1].t()).mm(self.H).mm(self.S_old[m])

                    else:
                        part4 = self.gamma * ((self.N - 1) ** (-2)) * self.H.mm(self.S_old[m - 1]).mm(
                            self.S_old[m - 1].t()).mm(self.H).mm(self.S_old[m])

                    self.S[m] = 0.5 * self.S_old[m] + 0.5 * self.S_old[m] * (Ws / (
                            self.theta[m].mm(self.S_old[m]) + self.lamda * self.S_old[m] + part3 + part4 + self.eps))

                    del part1_dot_divide_part2
                    del Ws
                    del part3
                    del part4

                self.C = 0.5 * self.C_old + 0.5 * self.C_old * Wc / (Dc + 2 * self.lamda * self.C_old + Tc + self.eps)

                del Tc
                del Wc
                del Dc

                F_tmp = torch.cat((self.C, self.S[0], self.S[1]), 1)

                if (j == (self.iter - 1)) or (torch.sum(torch.sum(torch.abs(F_tmp - F_old), dim=0)) < self.rho):
                    break
                else:
                    F_old = F_tmp.clone()
                    self.C_old = self.C.clone()
                    self.S_old = [self.S[0].clone(), self.S[1].clone()]
                
            del F_tmp
            del F_old
            del self.C_old
            del self.S_old

            for m in range(self.num_views):
                tmp = torch.cat((self.C, self.S[m]), 1)
                tmp = tmp.mm(tmp.t())
                score += -torch.sum(self.part1[m] * (torch.log(tmp + self.eps))) + torch.sum(
                    self.theta[m] * tmp) + self.lamda * (
                                 torch.norm(self.S[m], p="fro") ** 2) + self.beta * torch.sum(
                    torch.trace(self.C.t().mm(self.D[m]).mm(self.C)) + torch.trace(
                        self.S[m].t().mm(self.D[m]).mm(self.S[m])) - torch.trace(
                        self.C.t().mm(self.ap[m] - self.an[m]).mm(self.C)) - torch.trace(
                        self.S[m].t().mm(self.ap[m] - self.an[m]).mm(self.S[m]))
                )
            del tmp
            score += 2 * self.lamda * torch.norm(self.C, p="fro") ** 2 + self.gamma * ((self.N - 1) ** (-2)) * torch.trace(self.S[0].t().mm(self.H).mm(self.S[1]).mm(
                     self.S[1].t()).mm(self.H).mm(self.S[0]))
 
            if score < self.lowest_score:
                final_C = self.C.clone()
                final_S0 = self.S[0].clone()
                final_S1 = self.S[1].clone()
                self.lowest_score = score.clone()
            
            print('socre: {:.3f}'.format(self.lowest_score.data))

            del self.theta
            del self.part1
            del self.D

            del self.C
            del self.S

        print('\nDetecting protein comolex··· \n')

        F_star = torch.cat((final_C, final_S0, final_S1), 1)
        F_hat = torch.cat((final_C, final_S0, final_S1), 1)

        del final_C
        del final_S0
        del final_S1

        F_star[F_star >= self.tau] = 1
        F_star[F_star < self.tau] = 0

        C_star = F_star[:, 0:self.Kc]
        C_star = torch.unique(C_star[:, torch.sum(C_star, 0) > 2], dim=1)

        S_star = F_star[:, self.Kc:(self.Kc + self.num_views * self.Ks)]
        S_star_A = S_star[:, 0:self.Ks]
        S_star_A = torch.unique(S_star_A[:, torch.sum(S_star_A, 0) > 2], dim=1)

        S_star_B = S_star[:, self.Ks:]
        S_star_B = torch.unique(S_star_B[:, torch.sum(S_star_B, 0) > 2], dim=1)

        F_star = torch.cat((C_star, S_star_A, S_star_B), 1)
        F_star_A = torch.cat((C_star, S_star_A), 1)
        F_star_B = torch.cat((C_star, S_star_B), 1)

        del S_star

        return F_hat, F_star, F_star_A, F_star_B, C_star, S_star_A, S_star_B
