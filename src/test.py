import torch
from gplr import GPLR
import matplotlib.pyplot as plt


X_test=torch.load('./data/x_test.txt')
X=torch.load('./data/x.txt')
Y=torch.load('./data/y.txt')

# ガウス過程回帰モデル
gp=GPLR()
# 学習
pred_dist, low, up=gp.pred(X, Y, X_test, mode=True, model_path='./model/parameter.txt')

torch.save(pred_dist, './model/pred_dist.txt')
torch.save(low, './model/low.txt')
torch.save(up, './model/up.txt')