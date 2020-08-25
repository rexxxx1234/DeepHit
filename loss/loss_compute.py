import torch
import torch.nn.functional as F
from torch import nn

_EPSILON = 1e-08
_SIGMA1 = 0.1


# USER-DEFINED FUNCTIONS
def log(x):
    return torch.log(x + _EPSILON)


def div(x, y):
    return torch.div(x, (y + _EPSILON))


class LossCompute(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def loss_log_likelihood(self, inputs, k, mask1):
        I_1 = torch.sign(k).bool()

        # for uncenosred: log P(T=t,K=k|x)
        loss1 = torch.sum(torch.sum(mask1 * inputs, dim=2), dim=1)
        loss1 = I_1 * log(loss1)
        # loss1 = log(loss1[I_1])

        # for censored: log \sum P(T>t|x)
        loss2 = torch.sum(torch.sum(mask1 * inputs, dim=2), dim=1)
        loss2 = ~I_1 * log(loss2)
        # loss2 = log(loss2[~I_1])

        loss = -torch.mean(loss1 + loss2)
        return loss

    # LOSS-FUNCTION 2 -- Ranking loss
    def loss_ranking(self, inputs, k, t, mask2, num_event, num_category):

        eta = []
        for e in range(num_event):
            one_vector = torch.ones_like(t, dtype=torch.float32)
            I_2 = torch.diag((k == e + 1).float())  # indicator for event
            # event specific joint prob.
            # tmp_e = torch.reshape(inputs[:, e, :], (-1, num_category))
            tmp_e = inputs[:, e, :]

            # no need to divide by each individual dominator
            R = tmp_e @ mask2.T
            # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

            diag_R = torch.diagonal(R)
            # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
            R = diag_R.expand_as(I_2) - R
            # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})
            R = torch.transpose(R, 0, 1)

            T = F.relu(torch.sign(t.expand_as(I_2) - t.expand_as(I_2).T))
            # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

            # only remains T_{ij}=1 when event occured for subject i
            T = torch.matmul(I_2, T)

            tmp_eta = torch.mean(T * torch.exp(-R / _SIGMA1), dim=1, keepdim=True)

            eta.append(tmp_eta)
        # eta = torch.stack(eta, dim=1)  # stack referenced on subjects
        # eta = torch.mean(torch.reshape(eta, (-1, num_event)),
        #                  dim=1,
        #                  keepdim=True)
        eta = torch.cat(eta, dim=1).mean(1)

        LOSS_2 = torch.sum(eta)  # sum over num_Events

        return LOSS_2

    # LOSS-FUNCTION 3 -- Calibration Loss
    def loss_calibration(self, inputs, k, t, mask2):
        '''
        This loss is not used in the original paper & code.
        '''
        eta = []
        for e in range(self.num_Event):
            one_vector = torch.ones_like(t, dtype=torch.float32)
            I_2 = (k == e + 1).float()  # indicator for event
            tmp_e = torch.reshape(
                inputs[:, e, :],
                (-1, self.num_Category))  # event specific joint prob.

            r = torch.sum(
                tmp_e * mask2,
                dim=0)  # no need to divide by each individual dominator
            tmp_eta = torch.mean((r - I_2)**2, dim=1, keepdim=True)

            eta.append(tmp_eta)
        eta = torch.stack(eta, dim=1)  # stack referenced on subjects
        eta = torch.mean(torch.reshape(eta, (-1, self.num_Event)),
                         dim=1,
                         keepdim=True)

        LOSS_3 = torch.sum(eta)  # sum over num_Events

        return LOSS_3

    # def loss_total(self, inputs, k, t, mask1, mask2, a, b, c):
    def loss_total(self, inputs, batch, num_event, num_category):
        data, label, time, mask1, mask2 = batch

        LOSS_TOTAL = self.hparams.alpha * self.loss_log_likelihood(inputs, label, mask1) +\
            self.hparams.beta * self.loss_ranking(inputs, label, time, mask2, num_event, num_category)

        return LOSS_TOTAL
