import os
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('./figs'):
    os.mkdir('./figs')

loss=np.loadtxt('./model/loss.txt')

N=len(loss)

epoch=[i for i in range(N)]

plt.plot(epoch, loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./figs/loss.png')

plt.show()