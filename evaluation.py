import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score

class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None, e=None, task="idhp"):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.e = e
        self.true_ite = None

        if y_cf is not None and (mu0 is None or mu1 is None):
            mu0 = y * (1 - t) + y_cf * t
            mu1 = y_cf * (1 - t) + y * t

        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

        self.mu0 = mu0
        self.mu1 = mu1

        self.hasnan = False
        self.task = task

    def rmse_ite(self, ypred1, ypred0):
        if self.true_ite is None:
            return np.nan
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def abs_ate(self, ypred1, ypred0):
        if self.true_ite is None:
            return np.nan
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))


    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)

    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf))) if self.y_cf is not None else np.nan
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)

        if self.task == 'jobs':
            policy_risk = self.policy(ypred1, ypred0)
            return ite, ate, policy_risk

        elif self.task == 'twins':
            auc = self.auc(ypred1, ypred0)
            return ite, ate, auc
        else:
            pehe = self.pehe(ypred1, ypred0)
            return ite, ate, pehe

    def auc(self, ypred1, ypred0):
        y_label = np.concatenate((self.mu0, self.mu1), axis=0)
        y_label_pred = np.concatenate((ypred0, ypred1), axis=0)
        roc_auc = roc_auc_score(y_label, y_label_pred)

        return roc_auc

    def policy(self, pred_y_1, pred_y_0):
        e = self.e
        y_true = self.y[e==1]
        t_true = self.t[e==1]

        pred_y_0 = pred_y_0[e==1]
        pred_y_1 = pred_y_1[e==1]

        if len(pred_y_0) == 0 or len(pred_y_1) == 0:
            print("Empty Prediction of y")
            return 1.

        pred_policy = (pred_y_1 > pred_y_0).astype(float)
        pred_policy_t = np.mean(pred_policy)
        pred_policy_c = 1 - pred_policy_t

        if np.sum((pred_policy==1) * (t_true==1)) == 0:
            pred_y_t = 0.
        else:
            pred_y_t = np.mean(y_true[(pred_policy==1) * (t_true==1)])

        if np.sum((pred_policy==0) * (t_true==0)) == 0:
            pred_y_c = 0.
        else:
            pred_y_c = np.mean(y_true[(pred_policy==0) * (t_true==0)])

        risk = 1 - (pred_y_t*pred_policy_t + pred_y_c*pred_policy_c)
        return risk
