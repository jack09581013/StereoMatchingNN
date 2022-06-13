import CSPN.cspn as cspn
import GANet.GANet_small_deep as ganet_small_deep
import GANet.GANet_small as ganet_small
import GANet.GANet_deep as ganet_deep
import GANet.GANet_md
import GDFNet.GDFNet_mdc4
import GDFNet.GDFNet_md6
import GDFNet.GDFNet_mdc6
import GDFNet.GDFNet_mdc6f
import GDFNet.GDFNet_dc6
import GDFNet.GDFNet_dc6f
import PSMNet.model as psmnet
import os
import utils
import tools
import torch
import torch.nn.functional as F
import torch.nn as nn
import GANet.module
import GANet.function
import ganet_lib


class Profile:
    def __init__(self):
        os.makedirs(self.version_file_path(), exist_ok=True)
        assert '-' not in str(self)

    def load_model(self, max_disparity, version=None):
        self.model = self.get_model(max_disparity).cuda()
        self.max_disparity = max_disparity

        if version is None:
            print('Find latest version')
            version = utils.get_latest_version(self.version_file_path())

        if version is None:
            print('Can not find any version')
            version = 1
        else:
            print('Using version:', version)
            nn_file = self.model_file_name(version)

            if os.path.exists(nn_file):
                print('Load version model:', nn_file)
                self.model.load_state_dict(torch.load(nn_file))
            else:
                raise Exception(f'Cannot find neural network file: {nn_file}')

            version += 1

        return version, self.model

    def load_history(self, version=None):
        loss_history = {
            'train': [],
            'test': []
        }

        if version is None:
            print('Find latest version')
            version = utils.get_latest_version(self.version_file_path())

        if version is None:
            print('Can not find any version')
            version = 1
        else:
            print('Using version:', version)
            ht_file = self.history_file_name(version)

            if os.path.exists(ht_file):
                print('Load version history:', ht_file)
                loss_history = tools.load(ht_file)
            else:
                raise Exception(f'Cannot find history file: {ht_file}')

            version += 1

        return version, loss_history

    def save_version(self, model, history, version):
        torch.save(model.state_dict(), self.model_file_name(version))
        tools.save(history, self.history_file_name(version))

    def model_file_name(self, version):
        return os.path.join(self.version_file_path(), f'{self}-{version}.nn')

    def history_file_name(self, version):
        return os.path.join(self.version_file_path(), f'{self}-{version}.ht')

    def version_file_path(self):
        return f'../../model/{self}'

    def get_model(self, max_disparity):
        raise NotImplementedError()

    def train(self, X, Y, dataset):
        raise NotImplementedError()

    def eval(self, X, Y, dataset):
        raise NotImplementedError()

    def __str__(self):
        return type(self).__name__


class GANet_small(Profile):
    def get_model(self, max_disparity):
        return ganet_small.GANetSmall(max_disparity)


class CSPN(Profile):
    def get_model(self, max_disparity):
        return cspn.CSPN(max_disparity, 3, 20, 3)


class GANet_small_deep(Profile):
    def get_model(self, max_disparity):
        return ganet_small_deep.GANet_small_deep(max_disparity)

    def train(self, X, Y, dataset):
        Y = Y[:, 0, :, :]
        disp0, disp1, disp2 = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])

        mask = utils.y_mask(Y, self.max_disparity, dataset)
        loss0 = F.smooth_l1_loss(disp0[mask], Y[mask])
        loss1 = F.smooth_l1_loss(disp1[mask], Y[mask])
        loss2 = F.smooth_l1_loss(disp2[mask], Y[mask])
        epe_loss = utils.EPE_loss(disp2[mask], Y[mask])
        loss = 0.2 * loss0 + 0.6 * loss1 + loss2

        return loss, epe_loss

    def eval(self, X, Y, dataset, lr_check=False, max_disparity_diff=1.5):
        Y = Y[:, 0, :, :]
        if lr_check:
            cost0, cost1, disp_left = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])

            flip_x = X[:, 0:3, :, :].data.cpu().numpy()
            flip_y = X[:, 3:6, :, :].data.cpu().numpy()
            flip_x = torch.tensor(flip_x[..., ::-1].copy()).cuda()
            flip_y = torch.tensor(flip_y[..., ::-1].copy()).cuda()

            disp_right = self.model(flip_y, flip_x)[2]
            disp_right = disp_right.data.cpu().numpy()
            disp_right = - torch.tensor(disp_right[..., ::-1].copy()).cuda()

            ganet_lib.cuda_left_right_consistency_check(disp_left, disp_right, 1, max_disparity_diff)

            mask = utils.y_mask(Y, self.model.max_disparity, dataset)
            mask = mask & (disp_left != -1)
            epe_loss = utils.EPE_loss(disp_left[mask], Y[mask])
            error_sum = utils.error_rate(disp_left[mask], Y[mask], dataset)

            return {
                'error_sum': error_sum,
                'total_eval': mask.float().sum(),
                'epe_loss': epe_loss,
                'disp': disp_left.float(),
            }

        else:
            cost0, cost1, disp = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])
            mask = utils.y_mask(Y, self.max_disparity, dataset)
            epe_loss = utils.EPE_loss(disp[mask], Y[mask])
            error_sum = utils.error_rate(disp[mask], Y[mask], dataset)
            return {
                'error_sum': error_sum,
                'total_eval': mask.float().sum(),
                'epe_loss': epe_loss,
                'disp': disp.float(),
            }


class GANet_md(Profile):
    def get_model(self, max_disparity):
        return GANet.GANet_md.GANet_md(max_disparity)

    def train(self, X, Y, dataset):
        Y = Y[:, 0, :, :]
        disp0, disp1, disp2 = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])

        mask = utils.y_mask(Y, self.max_disparity, dataset)
        loss0 = F.smooth_l1_loss(disp0[mask], Y[mask])
        loss1 = F.smooth_l1_loss(disp1[mask], Y[mask])
        loss2 = F.smooth_l1_loss(disp2[mask], Y[mask])
        epe_loss = utils.EPE_loss(disp2[mask], Y[mask])
        loss = 0.2 * loss0 + 0.6 * loss1 + loss2

        return {
            'loss': loss,
            'epe_loss': epe_loss
        }

    def eval(self, X, Y, dataset):
        Y = Y[:, 0, :, :]
        disp = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])
        mask = utils.y_mask(Y, self.max_disparity, dataset)
        epe_loss = utils.EPE_loss(disp[mask], Y[mask])
        error_sum = utils.error_rate(disp[mask], Y[mask], dataset)
        return {
            'error_sum': error_sum,
            'total_eval': mask.float().sum(),
            'epe_loss': epe_loss,
            'disp': disp.float(),
        }


class GANet_deep(Profile):
    def get_model(self, max_disparity):
        return ganet_deep.GANet_deep(max_disparity)


class GDFNet_c(Profile):
    def get_model(self, max_disparity):
        self.disparity_class = GANet.module.DisparityClassRegression(max_disparity)
        self.disparity = GANet.module.DisparityRegression(max_disparity)
        self.squeeze_cost = GANet.module.SqueezeCost()
        self.squeeze_cost_grad = GANet.module.SqueezeCostByGradient()

    def train(self, X, Y, dataset):
        Y = Y[:, 0, :, :]

        cost0, cost1, cost2 = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])
        loss0 = self.disparity_class(cost0, Y)
        loss1 = self.disparity_class(cost1, Y)
        loss2 = self.disparity_class(cost2, Y)
        loss = 0.2 * loss0 + 0.6 * loss1 + loss2

        disp = torch.argmax(cost2, dim=1).float()

        mask = utils.y_mask(Y, self.max_disparity, dataset)
        epe_loss = utils.EPE_loss(disp[mask], Y[mask])

        return {
            'loss': loss,
            'epe_loss': epe_loss,
            'disp': disp
        }

    def eval(self, X, Y, dataset):
        Y = Y[:, 0, :, :]
        cost = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])

        disp = torch.argmax(cost, dim=1)
        cost = F.softmax(cost, dim=1)
        squeeze_mask, cost = self.squeeze_cost_grad(cost, disp.float())
        disp = self.disparity(cost)

        mask = utils.y_mask(Y, self.max_disparity, dataset)
        epe_loss = utils.EPE_loss(disp[mask], Y[mask])
        error_sum = utils.error_rate(disp[mask], Y[mask], dataset)

        return {
            'error_sum': error_sum,
            'total_eval': mask.float().sum(),
            'epe_loss': epe_loss,
            'cost': cost,
            'disp': disp.float(),
        }


class GDFNet_md6(Profile):
    def get_model(self, max_disparity):
        return GDFNet.GDFNet_md6.GDFNet_md6(max_disparity)

    def train(self, X, Y, dataset):
        Y = Y[:, 0, :, :]
        disp0, disp1, disp2 = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])

        mask = utils.y_mask(Y, self.max_disparity, dataset)
        loss0 = F.smooth_l1_loss(disp0[mask], Y[mask])
        loss1 = F.smooth_l1_loss(disp1[mask], Y[mask])
        loss2 = F.smooth_l1_loss(disp2[mask], Y[mask])
        epe_loss = utils.EPE_loss(disp2[mask], Y[mask])
        loss = 0.2 * loss0 + 0.6 * loss1 + loss2

        return {
            'loss': loss,
            'epe_loss': epe_loss
        }

    def eval(self, X, Y, dataset):
        Y = Y[:, 0, :, :]
        disp = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])
        mask = utils.y_mask(Y, self.max_disparity, dataset)
        epe_loss = utils.EPE_loss(disp[mask], Y[mask])
        error_sum = utils.error_rate(disp[mask], Y[mask], dataset)

        return {
            'error_sum': error_sum,
            'total_eval': mask.float().sum(),
            'epe_loss': epe_loss,
            'cost_left': None,
            'flip_cost': None,
            'cost_merge': None,
            'confidence_error': None,
            'disp': disp.float(),
        }

class GDFNet_mdc6(GDFNet_c):
    def get_model(self, max_disparity):
        GDFNet_c.get_model(self, max_disparity)
        return GDFNet.GDFNet_mdc6.GDFNet_mdc6(max_disparity)

    def train(self, X, Y, dataset):

        self.model.flip = False
        train_dict = super().train(X, Y, dataset)

        return train_dict

    def eval(self, X, Y, dataset, merge_cost=False, lr_check=False, candidate=False, regression=True, penalize=False,
             slope=1, max_disparity_diff=1.5):
        assert not (merge_cost and lr_check), 'do not use merge cost and lr check at the same time'
        assert not (candidate and lr_check), 'do not use candidate error rate and lr check at the same time'
        assert not self.model.training
        Y = Y[:, 0, :, :]

        # Calculate cost
        self.model.flip = False
        cost_left = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])

        if merge_cost or lr_check:
            self.model.flip = True
            cost_right = self.model(X[:, 0:3, :, :], X[:, 3:6, :, :])

        # Penalize cost
        if penalize:
            cost_process_left = penalize_cost_by_disparity(cost_left, slope)
            if merge_cost or lr_check:
                cost_process_right = penalize_cost_by_disparity(cost_right, slope)
        else:
            cost_process_left = cost_left
            if merge_cost or lr_check:
                cost_process_right = cost_right

        # Merge cost & Argmax disparity
        if merge_cost:
            cost_merge = cost_process_left.clone()
            flip_cost = GANet.function.FlipCost.apply(cost_process_right)
            confidence_error = disparity_confidence(cost_left, flip_cost)
            average_confidence_error = confidence_error[:, self.max_disparity:].mean()
            cost_merge[..., self.max_disparity:] = (cost_merge[..., self.max_disparity:] + flip_cost[...,
                                                                                           self.max_disparity:]) / 2
            disp_max_left = torch.argmax(cost_merge, dim=1).float()

        else:
            cost_merge = None
            flip_cost = None
            confidence_error = None
            average_confidence_error = None
            disp_max_left = torch.argmax(cost_process_left, dim=1).float()

        if lr_check:
            disp_max_right = torch.argmax(cost_process_right, dim=1).float()

        # Suppress Regression
        if regression:
            if merge_cost:
                cost = F.softmax(cost_merge, dim=1)
            else:
                cost = F.softmax(cost_left, dim=1)
            squeeze_mask, cost = self.squeeze_cost_grad(cost, disp_max_left)
            disp_left = self.disparity(cost)

            if lr_check:
                cost = F.softmax(cost_right, dim=1)
                cost = self.squeeze_cost_grad(cost, disp_max_right)[1]
                disp_right = self.disparity(cost)

        else:
            disp_left = disp_max_left
            if candidate:
                cost = F.softmax(cost_left, dim=1)
                squeeze_mask = self.squeeze_cost_grad(cost, disp_max_left)[0]
            if lr_check:
                disp_right = disp_max_right

        # Left-Right consistency check
        if lr_check:
            disp_right = disp_right.data.cpu().numpy()
            disp_right = - torch.tensor(disp_right[..., ::-1].copy()).cuda()
            ganet_lib.cuda_left_right_consistency_check(disp_left, disp_right, max_disparity_diff)

        # Evaluation
        mask = utils.y_mask(Y, self.max_disparity, dataset)
        if lr_check:
            mask &= (disp_left != -1)

        if candidate:
            if merge_cost:
                disp_max = torch.argmax(cost_merge * (squeeze_mask == 0), dim=1)
            else:
                disp_max = torch.argmax(cost_process_left * (squeeze_mask == 0), dim=1)

            if regression:
                if merge_cost:
                    cost = F.softmax(cost_merge, dim=1)
                else:
                    cost = F.softmax(cost_left, dim=1)

                cost = self.squeeze_cost_grad(cost, disp_max.float())[1]
                disp_left_2 = self.disparity(cost)
            else:
                disp_left_2 = disp_max

            disp = torch.cat([disp_left.unsqueeze(1), disp_left_2.unsqueeze(1)], dim=1)
            epe_loss0 = (disp[:, 0][mask] - Y[mask]).abs()
            epe_loss1 = (disp[:, 1][mask] - Y[mask]).abs()
            epe_loss = torch.min(epe_loss0, epe_loss1).mean()
            error_sum = utils.error_rate_candidate(disp, Y, dataset, mask)

        else:
            epe_loss = utils.EPE_loss(disp_left[mask], Y[mask])
            error_sum = utils.error_rate(disp_left[mask], Y[mask], dataset)

        return {
            'error_sum': error_sum,
            'total_eval': mask.float().sum(),
            'epe_loss': epe_loss,
            'cost_left': cost_left,
            'flip_cost': flip_cost,
            'cost_merge': cost_merge,
            'confidence_error': confidence_error,
            'CE_avg': average_confidence_error,
            'disp': disp_left.float(),
        }


class GDFNet_mdc6f(GDFNet_mdc6):
    def get_model(self, max_disparity):
        GDFNet_c.get_model(self, max_disparity)
        return GDFNet.GDFNet_mdc6f.GDFNet_mdc6f(max_disparity)

    def train(self, X, Y, dataset, flip=False):
        if flip:
            Y = Y[..., self.max_disparity:]

        self.model.flip = flip
        train_dict = GDFNet_c.train(self, X, Y, dataset)

        return train_dict


class GDFNet_mdc4(GDFNet_c):
    def get_model(self, max_disparity):
        GDFNet_c.get_model(self, max_disparity)
        return GDFNet.GDFNet_mdc4.GDFNet_mdc4(max_disparity)


class PSMNet(Profile):
    def get_model(self, max_disparity):
        return psmnet.PSMNet(max_disparity)

class GDFNet_dc6(GDFNet_mdc6):
    def get_model(self, max_disparity):
        GDFNet_c.get_model(self, max_disparity)
        return GDFNet.GDFNet_dc6.GDFNet_dc6(max_disparity)

class GDFNet_dc6f(GDFNet_mdc6f):
    def get_model(self, max_disparity):
        GDFNet_c.get_model(self, max_disparity)
        return GDFNet.GDFNet_dc6f.GDFNet_dc6f(max_disparity)

def penalize_cost_by_impossible(cost):
    impossible = torch.argmin(cost, dim=1).unsqueeze(1)
    cost_penalize = torch.arange(0, cost.size(1)).to(cost.device).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    cost_penalize = cost_penalize.repeat(cost.size(0), 1, cost.size(2), cost.size(3))
    cost_penalize = (cost_penalize - impossible).abs().float()
    cost_penalize = F.normalize(cost_penalize, dim=1, p=1)
    cost = cost - cost_penalize
    return cost


def penalize_cost_by_disparity(cost, p):
    cost = F.normalize(cost, dim=1, p=1)
    penalize = torch.arange(0, cost.size(1), dtype=torch.float).to(cost.device).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    penalize = F.normalize(penalize, dim=1, p=1) * p
    penalize = penalize.repeat(cost.size(0), 1, cost.size(2), cost.size(3))

    cost_penalize = (cost - penalize)
    cost_penalize = F.normalize(cost_penalize, dim=1, p=1)
    return cost_penalize


def disparity_confidence(cost, flip_cost):
    disp = torch.argmax(cost, dim=1).unsqueeze(1)
    mask = torch.zeros(cost.size(), dtype=torch.bool).to(cost.device)
    mask.scatter_(1, disp, 1)
    confidence = ((cost - flip_cost).abs() * mask).sum(dim=1)
    confidence[:, :, :cost.size(1)] = 0
    return confidence
