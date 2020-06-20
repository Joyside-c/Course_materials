#!/usr/bin/env python
# coding: utf-8

# ## lassolars
# ##### cueb-22019212043-程义淇(13654187167@163.com)

# In[1]:


import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cholesky
from numpy import mat,diag
import math
import time
import random

from itertools import chain
from math import log
import sys
import warnings

from scipy import linalg, interpolate
from scipy.linalg.lapack import get_lapack_funcs
from joblib import Parallel, delayed

from sklearn import linear_model
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.utils import arrayfuncs, as_float_array, check_X_y


# In[357]:


#生成数据
Xp=10
Xn=100
mean=np.zeros(Xp)

cov =np.eye(Xp)
for i in range(1,Xp):
    cov+=(np.eye(Xp,k=i)*0.9**i+np.eye(Xp,k=-i)*0.9**i)
    
x= np.random.multivariate_normal(mean, cov, Xn)
x_one=np.hstack((np.ones((Xn,1)),x))

residual=np.random.randn(Xn).reshape(Xn,1)

a=np.array([1]*int(Xp/2)).reshape(int(Xp/2),1)
b=-1*a
betaTrue=list(chain(*np.hstack((a,b))))#将数组展开成一列
betaTrue.append(1)
betaTrue=np.array(betaTrue).reshape(Xp+1,1)

y=np.dot(x_one,betaTrue)+residual


# In[361]:


#定义各种变量和集合
X=x
Gram=np.dot(X.T,X)
Cov = np.dot(X.T, y)
n_features = X.shape[1]
n_samples = X.shape[0]
#在return_path为True的条件下
coefs = np.zeros((n_features + 1, n_features))
alphas = np.zeros(n_features + 1)
n_iter, n_active = 0, 0
active, indices = list(), np.arange(n_features)
sign_active = np.empty(n_features, dtype=np.int8)
L = np.empty((n_features, n_features), dtype=Gram.dtype)
swap, nrm2 = linalg.get_blas_funcs(('swap', 'nrm2'), (Cov,))
solve_cholesky, = get_lapack_funcs(('potrs',), (L,))
tiny32 = np.finfo(np.float32).tiny  # to avoid division by 0 warning 
equality_tolerance = np.finfo(np.float32).eps
Gram_copy = Gram.copy()
Cov_copy = Cov.copy()
drop=False
SOLVE_TRIANGULAR_ARGS = {'check_finite': False}


# In[362]:


cplist=[]
while  True:
    #退出循环条件
    if n_active >= n_features:
        break
    #每一轮更新要进去的变量的序号
    C_idx=np.argmax(np.abs(Cov))#协方差最大的变量的下标
    C_ = Cov[C_idx]#绝对值最大的协方差
    C = np.fabs(C_)#绝对值最大的协方差的绝对值
    #新的α
    alpha = alphas[n_iter, np.newaxis]
    coef = coefs[n_iter]#参数矢量
    prev_alpha = alphas[n_iter - 1, np.newaxis]#扩展一个维度
    prev_coef = coefs[n_iter - 1]
    alpha[0] = C / n_samples
    
    #drop可控制要不要在损失函数达到最小时，剩下变量不再进入活跃集 为了显示所有变量迭代的效果全设置为False
    if not drop:
            ##########################################################
            # Append x_j to the Cholesky factorization of (Xa * Xa') #
            #                                                        #
            #            ( L   0 )                                   #
            #     L  ->  (       )  , where L * w = Xa' x_j          #
            #            ( w   z )    and z = ||x_j||                #
            #                                                        #
            ##########################################################
        sign_active[n_active] = np.sign(C_)#是否在活跃集中的标志
        m, n = n_active, C_idx + n_active#初始m=0,n为相关系数最大的变量序号
        Cov[C_idx], Cov[0] = swap(Cov[C_idx], Cov[0])
        indices[n], indices[m] = indices[m], indices[n]#m,n索引互换？
        Cov_not_shortened = Cov#保留原始协方差矩阵
        Cov = Cov[1:]  # remove Cov[0]
        
        Gram[m], Gram[n] = swap(Gram[m], Gram[n])
        Gram[:, m], Gram[:, n] = swap(Gram[:, m], Gram[:, n])
        c = Gram[n_active, n_active]
        L[n_active, :n_active] = Gram[n_active, :n_active]

        # 更新Gram的cholesky 分解
        if n_active:
            linalg.solve_triangular(L[:n_active, :n_active],
                                L[n_active, :n_active],
                                trans=0, lower=1,
                                overwrite_b=True,
                                **SOLVE_TRIANGULAR_ARGS)
        v = np.dot(L[n_active, :n_active], L[n_active, :n_active])
        diag = np.sqrt(np.abs(c - v))
        L[n_active, n_active] = diag
        #进活跃集
        active.append(indices[n_active])
        n_active += 1
        #
        least_squares, _ = solve_cholesky(L[:n_active, :n_active],
                                          sign_active[:n_active],
                                          lower=True)
        if least_squares.size == 1 and least_squares == 0:
            #sign_active[:n_active] = 0时的情况
            least_squares[...] = 1
            AA = 1.
        else:
            AA = 1. / np.sqrt(np.sum(least_squares * sign_active[:n_active]))
            least_squares *= AA
        corr_eq_dir = np.dot(Gram[:n_active, n_active:].T,
                                 least_squares)
        #在相关系数正负两种情况下有两个γ，选正的最小值
        g1 = arrayfuncs.min_pos((C - Cov) / (AA - corr_eq_dir + tiny32))
        g2 = arrayfuncs.min_pos((C + Cov) / (AA + corr_eq_dir + tiny32))
        gamma_ = min(g1, g2, C / AA)
        #change names for these variables: z
        z = -coef[active] / (least_squares + tiny32)
        z_pos = arrayfuncs.min_pos(z)
        if z_pos < gamma_:
            # some coefficients have changed sign
            idx = np.where(z == z_pos)[0][::-1]

            # 更新sign
            sign_active[idx] = -sign_active[idx]

            gamma_ = z_pos
            #drop = True

        n_iter += 1
        if n_iter >= coefs.shape[0]:
            del coef, alpha, prev_alpha, prev_coef
            # resize the coefs and alphas array
            add_features = 2 * max(1, (n_features - n_active))
            coefs = np.resize(coefs, (n_iter + add_features, n_features))
            coefs[-add_features:] = 0
            alphas = np.resize(alphas, n_iter + add_features)
            alphas[-add_features:] = 0
        coef = coefs[n_iter]
        prev_coef = coefs[n_iter - 1]
        coef[active] = prev_coef[active] + gamma_ * least_squares
        # update correlations
        a=gamma_ * corr_eq_dir
        Cov -= a[:, np.newaxis]
        # See if any coefficient has changed sign
        if drop :

            # handle the case when idx is not length of 1
            for ii in idx:
                arrayfuncs.cholesky_delete(L[:n_active, :n_active], ii)

            n_active -= 1
            # handle the case when idx is not length of 1
            drop_idx = [active.pop(ii) for ii in idx]


            for ii in idx:
                for i in range(ii, n_active):
                    indices[i], indices[i + 1] = indices[i + 1], indices[i]
                    Gram[i], Gram[i + 1] = swap(Gram[i], Gram[i + 1])
                    Gram[:, i], Gram[:, i + 1] = swap(Gram[:, i],
                                                      Gram[:, i + 1])
            temp = Cov_copy[drop_idx] - np.dot(Gram_copy[drop_idx], coef)
            Cov = np.r_[temp, Cov]

            sign_active = np.delete(sign_active, idx)
            sign_active = np.append(sign_active, 0.)  # just to maintain size
    alphas = alphas[:n_iter + 1]
    coefs = coefs[:n_iter + 1]
    #cp准则 用平方损失 加模型复杂度的惩罚op
    cp=np.sum((np.dot(X, coef)[:,np.newaxis]-y)**2)/Xn+2*sum(np.dot(np.dot(X, coef),y))/Xn
    cplist.append(cp)
    #print(n_iter,active)
    print("循环次数",n_iter,", α",alphas,", 活跃集",active,", cp值",cp,", cp值中加的op大小",2*sum(np.dot(np.dot(X, coef),y))/Xn )
    print("选入变量对应系数",coefs.T)#coefs.T


# In[363]:


cplist#变量相关度很高 只选一个变量的误差是最小的


# In[ ]:




