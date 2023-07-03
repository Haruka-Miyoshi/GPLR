import torch
from gplr import GPLR

X=torch.load('./data/x.txt')
Y=torch.load('./data/y.txt')

# ガウス過程回帰モデル
gp=GPLR()
# 学習
gp.fit(X, Y, mode=True)