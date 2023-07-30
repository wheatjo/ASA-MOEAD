from pymoo.core.problem import Problem
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.interpolate import RBFInterpolator
from pymoo.problems.single.sphere import Sphere
from ezmodel.models.rbf import RBF


class SurrogateModel(object):

    def __init__(self, n_var, n_obj, n_ieq_constr):
        super(SurrogateModel, self).__init__()
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_ieq_constr = n_ieq_constr

    def fit(self, X: np.ndarray, F: np.ndarray, G: np.ndarray):
        """
        Fit the surrogate model from data (X, F, G)
        :param X:
        :param F:
        :param G:
        :return: None
        """
        pass

    def evaluate(self, X: np.ndarray):
        pass


class SurrogateProblem(Problem):

    def __init__(self, n_var, surr_model: SurrogateModel, xl: float, xu: float, n_obj=1, n_ieq_constr=1):
        super(SurrogateProblem, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.surr_model = surr_model

    def _evaluate(self, x, out, *args, **kwargs):
        pred_res = self.surr_model.evaluate(x)
        out['F'] = pred_res['F']
        out['G'] = pred_res['G']


class RBFModel(SurrogateModel):

    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn


    def __init__(self, n_var, n_obj, n_ieq_constr, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr)

        self.eps = 1e-8
        self.F_rbf = []
        self.G_rbf = []
        for i in range(n_obj):
            self.F_rbf.append(RBF(kernel="cubic"))

        for i in range(n_ieq_constr):
            self.G_rbf.append(RBF(kernel="cubic"))

    def fit(self, X: np.ndarray, F: np.ndarray, G: np.ndarray):

        for i in range(self.n_obj):
            print('n_obj', i)
            self.F_rbf[i].fit(X, F[:, i])

        for i in range(self.n_ieq_constr):
            print('n_con', i)
            self.G_rbf[i].fit(X, G[:, i])

    # def fit(self, X: np.ndarray, F: np.ndarray, G: np.ndarray):
    #     F_rbf_temp = []
    #     G_rbf_temp = []
    #     for i in range(self.n_obj):
    #         print('n_obj', i)
    #         rbf = RBFInterpolator(X, F[:, i], kernel="cubic")
    #         F_rbf_temp.append(rbf)
    #
    #     for i in range(self.n_ieq_constr):
    #         print('n_con', i)
    #         rbf = RBFInterpolator(X, G[:, i], kernel="cubic")
    #         G_rbf_temp.append(rbf)
    #
    #     self.F_rbf = F_rbf_temp
    #     self.G_rbf = G_rbf_temp


    def evaluate(self, X: np.ndarray) -> dict:
        F = []
        G = []

        for rbf_model in self.F_rbf:
            f_hat = rbf_model.predict(X)[:, 0]
            F.append(f_hat)

        for rbf_model in self.G_rbf:
            g_hat = rbf_model.predict(X)[:, 0]
            G.append(g_hat)

        F_pred = np.stack(F, axis=1)
        G_pred = np.stack(G, axis=1)

        out = {'F': F_pred, 'G': G_pred}
        return out


class SurrProblemFDLS(Problem):

    def __init__(self, n_var, n_obj, n_ieq_constr, surr_model: RBFModel, xl: float, xu: float, ref_inner: np.ndarray,
                 ref_outer: np.ndarray):
        super(SurrProblemFDLS, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.surr_model = surr_model
        self.ref_inner = ref_inner
        self.ref_outer = ref_outer

    def _evaluate(self, x, out, *args, **kwargs):
        pred_res = self.surr_model.evaluate(x)
        pred_G = pred_res['G']
        pred_F = pred_res['F']

        cv = np.sum(np.maximum(pred_G, 0.0), axis=1)
        fea_flag = cv == 0

        anoc = np.ones(pred_G.shape[0])
        infea_cv = cv[~fea_flag]
        fea_cv = np.sum(pred_G[fea_flag], axis=1)
        anoc[~fea_flag] = infea_cv
        anoc[fea_flag] = fea_cv
        if pred_F.shape == (50, 4):
            print(pred_F)
        ac_inner = calc_d2s(pred_F, self.ref_inner)
        ac_outer = calc_d2s(pred_F, self.ref_outer)
        ac = ac_inner - ac_outer

        out['F'] = ac
        out['G'] = anoc


class SurrProblemCDLS(Problem):

    def __init__(self, n_var, n_obj, n_ieq_constr, surr_model: RBFModel, xl: float, xu: float, ref, decomposition):
        super(SurrProblemCDLS, self).__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, xl=xl, xu=xu)
        self.surr_model = surr_model
        self.ref = ref
        self.decomposition = decomposition

    def _evaluate(self, x, out, *args, **kwargs):
        pred_res = self.surr_model.evaluate(x)
        pred_G = pred_res['G']
        pred_F = pred_res['F']

        deco_F = self.decomposition.do(pred_F, self.ref)
        out['F'] = deco_F
        out['G'] = pred_G



def calc_d2s(F, ref_dirs):
    ref_dirs_norm = np.linalg.norm(ref_dirs, axis=1)
    d1s = F.dot(ref_dirs.T) / ref_dirs_norm
    F_norm = np.linalg.norm(F, axis=1).reshape(-1, 1)
    d2s = np.sqrt(F_norm**2 - d1s**2)
    d2s_min = np.min(d2s, axis=1)
    return d2s_min



