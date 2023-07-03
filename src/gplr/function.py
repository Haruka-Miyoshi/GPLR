import os
import torch
import gpytorch
import numpy as np
from .model import *

"""ガウス過程回帰モデル"""
class GPLR(object):
    """initialize"""
    def __init__(self):
        # デバイス設定 GPU or CPU
        self.__device="cuda" if torch.cuda.is_available() else "cpu"
        # モデル
        self.__model=None
        # 尤度P(y|f,sig^2)->ガウス分布
        self.__likelihood=gpytorch.likelihoods.GaussianLikelihood()
        # 最適化アルゴリズム
        self.__opt=None
        # 周辺対数尤度計算アルゴリズム
        self.__mll=None
        # 学習係数
        self.__lr=0.1
        # 損失関数
        self.__loss_hist=None

        # save file path
        self.FILE_PATH=os.path.join('./model')
        # フォルダを生成
        if not os.path.exists(self.FILE_PATH):
            os.mkdir(self.FILE_PATH)
    
    """学習"""
    def fit(self, x, y, epoch=200, mode=False):
        # GPU領域に変換
        x=x.to(device=self.__device)
        y=y.to(device=self.__device)

        # 学習モデル
        self.__model=Model(x,y, self.__likelihood).to(device=self.__device)
        # 最適化アルゴリズム
        self.__opt=torch.optim.Adam(self.__model.parameters(), lr=self.__lr)
        # 周辺対数尤度 計算
        self.__mll=gpytorch.mlls.ExactMarginalLogLikelihood(self.__likelihood, self.__model)
        # 学習毎の損失
        self.__loss_hist=torch.zeros(epoch)

        # 学習モードに切り替え
        self.__model.train()
        self.__likelihood.train()

        # 学習
        for n in range(epoch):
            # 勾配を0に初期化
            self.__opt.zero_grad()
            # 予測値
            y_hat=self.__model(x)
            # 損失計算
            loss=-self.__mll(y_hat, y)
            # 勾配計算
            loss.backward()
            # パラメータを更新
            self.__opt.step()
            # 損失計算
            self.__loss_hist[n]=loss.item()
        
        if mode:
            # パラメータ保存名
            file_name='parameter.txt'
            # パラメータ保存
            PARAM_SAVE=os.path.join(self.FILE_PATH, file_name)
            # 学習したパラメータを保存
            torch.save(self.__model.state_dict(), PARAM_SAVE)
            # パラメータ保存名
            file_name='loss.txt'
            # パラメータ保存
            LOSS_SAVE=os.path.join(self.FILE_PATH, file_name)
            # 学習したパラメータを保存
            np.savetxt(LOSS_SAVE, self.__loss_hist)
    

    """予測"""
    def pred(self, x, y, x_test, mode=True, model_path=''):
        # GPU領域に変換
        x=x.to(device=self.__device)
        y=y.to(device=self.__device)

        # 学習モデル
        self.__model=Model(x, y, self.__likelihood).to(device=self.__device)
        # 学習済みモデル
        if mode:
            # 学習済みモデル読み込み
            self.__model.load_state_dict(torch.load(model_path))

        self.__model.eval()
        self.__likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # 予測分布を生成
            pred_dist=self.__likelihood(self.__model(x_test))
            # 信用区間を生成
            low, up = pred_dist.confidence_region()
        
        return pred_dist, low, up