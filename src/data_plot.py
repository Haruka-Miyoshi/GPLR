import matplotlib.pyplot as plt
import torch
import os

if not os.path.exists('./figs'):
    os.mkdir('./figs')

X=torch.load('./data/x.txt')
Y=torch.load('./data/y.txt')

plt.scatter(X,Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('./figs/data.png')
plt.show()