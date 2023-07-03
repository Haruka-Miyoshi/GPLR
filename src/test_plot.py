import os
import torch
import matplotlib.pyplot as plt

if not os.path.exists('./figs'):
    os.mkdir('./figs')

X=torch.load('./data/x.txt')
Y=torch.load('./data/y.txt')
X_test=torch.load('./data/x_test.txt')

pred_dist=torch.load('./model/pred_dist.txt')
low=torch.load('./model/low.txt')
up=torch.load('./model/up.txt')

fig, ax=plt.subplots(1, 1, figsize=(10, 4))
ax.plot(X.numpy(), Y.numpy(), 'k*', label='Observed Data')
ax.plot(X_test.numpy(), pred_dist.mean.numpy(), 'b', label='Mean')
ax.fill_between(X_test.numpy(), low.numpy(), up.numpy(), alpha=0.5, label='Confidence')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.legend()
plt.savefig('./figs/test_plot.png')
plt.show()