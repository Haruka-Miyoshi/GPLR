import gpytorch
from gpytorch.models import ExactGP

"""ガウス過程回帰モデル"""
class Model(gpytorch.models.ExactGP):
    """initialize"""
    def __init__(self, x, y, likelihood):
        super(Model, self).__init__(x, y, likelihood)

        # 平均関数
        self.__mean=gpytorch.means.ConstantMean()
        # カーネル関数(共分散行列) -> RBF -> exp(-|x-x'|^2/theta)
        self.__convar=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    """ガウス過程 生成過程"""
    def forward(self, x):
        # 入力xに対する平均関数を計算
        mean_x=self.__mean(x)
        # 入力xに対する共分散行列を計算 -> 入力が近いカーネル関数の分布を生成
        convar_x=self.__convar(x)
        # 多変量正規分布から関数をサンプリング
        return gpytorch.distributions.MultivariateNormal(mean_x, convar_x)