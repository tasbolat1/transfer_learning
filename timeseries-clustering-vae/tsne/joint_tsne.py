'''
ref source: https://blog.csdn.net/leida_wt/article/details/84993848

RH: based on tsne, create class called myJointTSNE which adds KL divergence between two datasets
'''

import numpy as np
import math
import matplotlib.pyplot as plt
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys
# import tsne

# 定义X data_num * fearures 原始
# 定义Y data_num * fearures 降维后


class myTSNE:
    def __init__(self, X, perp=30, seed=1):
        '''类初始化

        Arguments:
            X {Tensor} -- 欲降维数据(n,nfeature)

        Keyword Arguments:
            perp {int} -- 困惑度 (default: {30})
        '''

        self.X = X
        self.N = X.shape[0]
        # 注意此处先在不要求grad时乘好常数再打开grad以保证其是叶子节点
        t.manual_seed(1)
        self.Y = (t.randn(self.N, 2)*1e-4).requires_grad_()
        self.perp = perp

    def cal_distance(self, data):
        '''计算欧氏距离 (calculate pairwise distances in batch)
        https://stackoverflow.com/questions/37009647/

        Arguments:
            data {Tensor} -- N*features

        Returns:
            Tensor -- N*N 距离矩阵，D[i,j]为distance(data[i],data[j])
        '''

        assert data.dim() == 2, 'should be N*features'
        r = (data*data).sum(dim=1, keepdim=True) # *:elementwise multiplication
        D = r-2*data@data.t()+r.t() # @: matrix multiplication
        return D

    def Hbeta(self, D, beta=1.0):
        '''计算给定某一行(n,)与sigma的pj|i与信息熵H

        Arguments:
            D {np array} -- 距离矩阵的i行，不包含与自己的，大小（n-1,)

        Keyword Arguments:
            beta {float} -- 即1/(2sigma^2) (default: {1.0})

        Returns:
            (H,P) -- 信息熵 , 概率pj|i
        '''

        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta)
        sumP = sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def p_j_i(self, distance_matrix, tol=1e-5, perplexity=30):
        '''由距离矩阵计算p(j|i)矩阵，应用二分查找寻找合适sigma

        Arguments:
            distance_matrix {np array} -- 距离矩阵(n,n)

        Keyword Arguments:
            tol {float} -- 二分查找允许误差 (default: {1e-5})
            perplexity {int} -- 困惑度 (default: {30})

        Returns:
            np array -- p(j|i)矩阵
        '''

        print("Computing pairwise distances...")
        (n, d) = self.X.shape
        D = distance_matrix
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        # 遍历每一个数据点
        for i in range(n):

            if i % 500 == 0:
                print("Computing P-values for point %d of %d..." % (i, n))

            # 准备Di，
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] # r_: rowwise merging
            (H, thisP) = self.Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries = 0
            # 开始二分搜索，直到满足误差要求或达到最大尝试次数
            while np.abs(Hdiff) > tol and tries < 50:

                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                (H, thisP) = self.Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # 最后将算好的值写至P，注意pii处为0
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

        print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P

    def cal_P(self, data, sigma=None):
        '''计算对称相似度矩阵

        Arguments:
            data {Tensor} - - N*features

        Keyword Arguments:
            sigma {Tensor} - - N个sigma(default: {None})

        Returns:
            Tensor - - N*N
        '''
        distance = self.cal_distance(data)  # 计算距离矩阵
        P = self.p_j_i(distance.numpy(), perplexity=self.perp)  # 计算原分布概率矩阵
        P = t.from_numpy(P).float()  # p_j_i为numpy实现的，这里变回Tensor
        P = (P + P.t())/P.sum()  # 对称化
        P = P * 4.  # 夸张
        P = t.max(P, t.tensor(1e-12))  # 保证计算稳定性
        return P

    def cal_Q(self, data):
        '''计算降维后相似度矩阵

        Arguments:
            data {Tensor} - - Y, N*2

        Returns:
            Tensor - - N*N
        '''

        Q = (1.0+self.cal_distance(data))**-1
        # 对角线强制为零
        Q[t.eye(self.N, self.N, dtype=t.long) == 1] = 0
        Q = Q/Q.sum()
        Q = t.max(Q, t.tensor(1e-12))  # 保证计算稳定性
        return Q

    def train(self, epoch=1000, lr=10, weight_decay=0, momentum=0.9, show=False, savefig=False, figname=None):
        '''
        Training process

        Keyword Arguments:
            epoch {int} -- 迭代次数 (default: {1000})
            lr {int} -- 学习率，典型10-100 (default: {10})
            weight_decay {int} -- L2正则系数 (default: {0})
            momentum {float} -- 动量 (default: {0.9})
            show {bool} -- 是否显示训练信息 (default: {False})

        Returns:
            Tensor -- 降维结果(n,2)
        '''

        # 先算出原分布的相似矩阵
        P = self.cal_P(self.X)
        optimizer = optim.SGD(
            [self.Y],
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
        loss_his = []
        print('training started @lr={},epoch={},weight_decay={},momentum={}'.format(
            lr, epoch, weight_decay, momentum))
        for i in range(epoch):
            if i % 100 == 0:
                print('running epoch={}'.format(i))
            if epoch == 100:
                P = P/4.0  # 100轮后取消夸张
            optimizer.zero_grad()
            Q = self.cal_Q(self.Y)
            loss = (P*t.log(P/Q)).sum()
            loss_his.append(loss.item())
            loss.backward()
            optimizer.step()
        print('train complete!')
        if savefig:
            assert show, "must opt in \"show\" to use savefig"
        if show:
            print('final loss={}'.format(loss_his[-1]))
            plt.plot(np.log10(loss_his))
            loss_his = []
            if savefig:
                assert figname is not None, "please specify a figname"
                plt.savefig(figname)
            plt.show()
        return self.Y.detach()


def any_Hbeta(D, beta=1.0):
    '''计算给定某一行(n,)与sigma的pj|i与信息熵H

    Arguments:
        D {np array} -- 距离矩阵的i行，不包含与自己的，大小（n-1,)

    Keyword Arguments:
        beta {float} -- 即1/(2sigma^2) (default: {1.0})

    Returns:
        (H,P) -- 信息熵 , 概率pj|i
    '''

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def any_p_j_i(n, d, distance_matrix, tol=1e-5, perplexity=30):
    '''由距离矩阵计算p(j|i)矩阵，应用二分查找寻找合适sigma

    Arguments:
        n: number of data points
        d: dimension of each data point
        distance_matrix {np array} -- 距离矩阵(n,n)

    Keyword Arguments:
        tol {float} -- 二分查找允许误差 (default: {1e-5})
        perplexity {int} -- 困惑度 (default: {30})

    Returns:
        np array -- p(j|i)矩阵
    '''

    print("Computing pairwise distances...")
    D = distance_matrix
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # 遍历每一个数据点
    for i in range(n):

        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # 准备Di，
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] # r_: rowwise merging
        (H, thisP) = any_Hbeta(Di, beta[i])
        Hdiff = H - logU
        tries = 0
        # 开始二分搜索，直到满足误差要求或达到最大尝试次数
        while np.abs(Hdiff) > tol and tries < 50:

            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            (H, thisP) = any_Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # 最后将算好的值写至P，注意pii处为0
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def cal_any_distance(data1, data2):
    '''计算欧氏距离 (calculate pairwise distances in batch)
    https://stackoverflow.com/questions/37009647/

    Arguments:
        data {Tensor} -- N*features

    Returns:
        Tensor -- N*N 距离矩阵，D[i,j]为distance(data[i],data[j])
    '''

    assert data1.dim() == 2 and data2.dim() == 2 , 'should be N*features'
    r = (data1*data2).sum(dim=1, keepdim=True) # *:elementwise multiplication
    # c=(a-b)**2 formula for matrix
    # c = a@a.T - a@b.T - b@a.T + b@b.T
    D = data1@data1.T - data1@data2.T -data2@data1.T + data2@data2.T
    return D    

def cal_any_P(data1, data2, sigma=None, perp=30):
    '''计算对称相似度矩阵

    Arguments:
        data {Tensor} - - N*features

    Keyword Arguments:
        sigma {Tensor} - - N个sigma(default: {None})

    Returns:
        Tensor - - N*N
    '''
    distance = cal_any_distance(data1, data2)  # 计算距离矩阵
    P = any_p_j_i(data1.shape[0], data1.shape[1], distance.numpy(), perplexity=perp)  # 计算原分布概率矩阵
    P = t.from_numpy(P).float()  # p_j_i为numpy实现的，这里变回Tensor
#     P = (P + P.t())/P.sum()  # 对称化
    P = P * 4.  # 夸张
    P = t.max(P, t.tensor(1e-13))  # 保证计算稳定性
    return P


def cal_any_Q(data1, data2):
    '''计算降维后相似度矩阵

    Arguments:
        data {Tensor} - - Y, N*2

    Returns:
        Tensor - - N*N
    '''

    Q = (1.0+cal_any_distance(data1, data2))**-1
#     # 对角线强制为零
#     Q[t.eye(self.N, self.N, dtype=t.long) == 1] = 0
    Q = Q/Q.sum()
    Q = t.max(Q, t.tensor(1e-12))  # 保证计算稳定性
    return Q


class myJointTSNE:
    def __init__(self, X1, X2, perp=30, seed=1):
        '''
        initialization
        
        Arguments:
            X1 {Tensor} -- (N, num_feature)
            X2 {Tensor} -- {N, num_feature}
            
            Note: X1 and X2 should have one-to-one mapping, or at least same class label, so KL divergence makes sense
        
        Keyword Arguments:
            perp {int} -- perplexity (typical 5-50, default 30)
        '''
        
        self.perp = perp
        self.X1 = X1
        self.X2 = X2
        assert X1.shape[0] == X2.shape[0], "X1 and X2 should have same shape, now {}, {}".format(X1.shape, X2.shape)
        self.N = X1.shape[0]
        self.seed = seed
        torch.manual_seed(self.seed)
        # randomly initialize target space
#         self.Y1 = (torch.randn(self.N, 2)*1e-4).requires_grad()
#         self.Y2 = (torch.randn(self.N, 2)*1e-4).requires_grad()
        self.t1 = myTSNE(self.X1, perp=self.perp, seed=self.seed)
        self.t2 = myTSNE(self.X2, perp=self.perp, seed=self.seed)
    
    def train(self, epoch=1000, lr=10, weight_decay=0, momentum=0.9, show=False):
        ''' 
        Training progress
        supercede train() funtion of myTSNE and add KL divergence 
        '''
        P1 = self.t1.cal_P(self.X1)
        P2 = self.t2.cal_P(self.X2)
        P3 = cal_any_P(self.X1, self.X2)
        optimizer = optim.SGD([self.t1.Y, self.t2.Y], lr=lr,
                              weight_decay=weight_decay, momentum=momentum)
        loss_his = []
        loss1_his = []
        loss2_his = []
        loss3_his = []
        loss4_his = []
        joint_loss = nn.L1Loss()
        print('training started @lr={},epoch={},weight_decay={},momentum={}'.format(
            lr, epoch, weight_decay, momentum))
        for i in range(epoch):
            if i % 100 == 0:
                print("running epoch={}".format(i))
            if epoch == 100:
                # stop exaggeration
                P1 = P1/4.0
                P2 = P2/4.0
                P3 = P3/4.0
            optimizer.zero_grad()
            Q1 = self.t1.cal_Q(self.t1.Y)
            Q2 = self.t2.cal_Q(self.t2.Y)
            Q3 = cal_any_Q(self.t1.Y, self.t2.Y)
#             print(P1.shape, Q1.shape) # (192, 192)
            loss1 = (P1*t.log(P1/Q1)).sum()
            loss2 = (P2*t.log(P2/Q2)).sum()
            loss3 = (P3*t.log(P3/Q3)).sum()
            loss4 = joint_loss(self.t1.Y, self.t2.Y)
#             print("ep {} losses: {}, {}, {}, {}".format(i, loss1, loss2, loss3, loss4))
            if i < 100:
                loss = loss1+loss2
            else:
                loss = loss1 + loss2 + 0.3* loss4 #0.0001*loss3 + 0*loss4
            loss_his.append(loss.item())
            loss1_his.append(loss1.item())
            loss2_his.append(loss2.item())
            loss3_his.append(loss3.item())
            loss4_his.append(loss4.item())
            loss.backward()
            optimizer.step()
#         print("training complete!")
        print("seed: {}, final loss = {}".format(self.seed, loss_his[-1]))
        if show:
            fig = plt.figure()
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            ax3 = fig.add_subplot(233)
            ax4 = fig.add_subplot(234)
            ax5 = fig.add_subplot(235)
            ax1.title.set_text('Total loss')
            ax1.plot(np.log10(loss_his))
            ax2.title.set_text('Loss1')
            ax2.plot(np.log10(loss1_his))
            ax3.title.set_text('Loss2')
            ax3.plot(np.log10(loss2_his))
            ax4.title.set_text('Loss3')
            ax4.plot(np.log10(loss3_his))
            ax5.title.set_text('Loss4')
            ax5.plot(np.log10(loss4_his))            
            plt.show()

#             plt.plot(np.log10(loss_his))
#             loss_his = []
#             plt.show()
        return self.t1.Y.detach(), self.t2.Y.detach()
            
    
    
def main():
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    X = t.load('./x.pt').float()  # 0,1二值化的
    X = t.from_numpy(PCA(n_components=30).fit_transform(X.numpy())).float()
    C = t.load('./labels.pt').float()
    X = X[0:600]
    C = C[0:600]

    T = myTSNE(X)
    res = T.train(epoch=600, lr=50, weight_decay=0.,
                  momentum=0.5, show=True).numpy()
    plt.scatter(res[:, 0], res[:, 1], c=C.numpy())
    plt.show()


if __name__ == '__main__':
    main()
