import os
import torch

if not os.path.exists('./data'):
    os.mkdir('./data')

# 真の関数
def true_func(x):
    return 1.5*torch.exp(x) * torch.sin(2*torch.pi * x)

# データ数
N=15
# 説明変数
X=torch.FloatTensor(N).uniform_(-1, 1)

# 目的変数
Y=true_func(X)+0.1*torch.randn(N)

# テストデータ
x_test=torch.linspace(-1,1,50)

torch.save(X, './data/x.txt')
torch.save(Y, './data/y.txt')
torch.save(x_test, './data/x_test.txt')