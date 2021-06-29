import torch
import torch.nn as nn
import numpy as np

class TensorAxis:
    N = 0
    H = 1
    W = 2
    C = 3


class CSFlow:
    def __init__(self, sigma=float(0.1), b=float(1.0)):
        self.b = b
        self.sigma = sigma

    def __calculate_CS(self, scaled_distances, axis_for_normalization=TensorAxis.C):
        self.scaled_distances = scaled_distances
        self.cs_weights_before_normalization = torch.exp((self.b - scaled_distances) / self.sigma)
        # self.cs_weights_before_normalization = 1 / (1 + scaled_distances)
        self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)
        # self.cs_NHWC = self.cs_weights_before_normalization

    # def reversed_direction_CS(self):
    #     cs_flow_opposite = CSFlow(self.sigma, self.b)
    #     cs_flow_opposite.raw_distances = self.raw_distances
    #     work_axis = [TensorAxis.H, TensorAxis.W]
    #     relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
    #     cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
    #     return cs_flow_opposite

    # --
    @staticmethod
    def create_using_L2(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        # Nx(HW)xC
        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        # Nx(HW)
        r_Ts = torch.sum(Tvecs * Tvecs, 2)
        r_Is = torch.sum(Ivecs * Ivecs, 2)
        raw_distances_list = []
        for i in range(sT[0]):
            # (HW)xC, (HW)XC, (HW), (HW)
            Ivec, Tvec, r_T, r_I = Ivecs[i], Tvecs[i], r_Ts[i], r_Is[i]
            # (HW)x(HW)
            A = Tvec @ torch.transpose(Ivec, 0, 1)  # (matrix multiplication)
            cs_flow.A = A
            # A = tf.matmul(Tvec, tf.transpose(Ivec))
            # (HW)x1
            r_T = torch.reshape(r_T, [-1, 1])  # turn to column vector
            # (HW)x(HW)   
            dist = r_T - 2 * A + r_I
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_L1(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        sT = T_features.shape
        sI = I_features.shape

        Ivecs = torch.reshape(I_features, (sI[0], -1, sI[3]))
        Tvecs = torch.reshape(T_features, (sI[0], -1, sT[3]))
        raw_distances_list = []
        for i in range(sT[0]):
            Ivec, Tvec = Ivecs[i], Tvecs[i]
            dist = torch.abs(torch.sum(Ivec.unsqueeze(1) - Tvec.unsqueeze(0), dim=2))
            dist = torch.reshape(torch.transpose(dist, 0, 1), shape=(1, sI[1], sI[2], dist.shape[0]))
            # protecting against numerical problems, dist should be positive
            dist = torch.clamp(dist, min=float(0.0))
            # dist = tf.sqrt(dist)
            raw_distances_list += [dist]

        cs_flow.raw_distances = torch.cat(raw_distances_list)

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    # --
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma=float(0.5), b=float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # prepare feature before calculating cosine distance
        cs_flow.center_by_T(T_features, I_features)
        CSFlow.l2_normalize_channelwise_inplace(cs_flow)
        # T_features, I_features = cs_flow.center_by_T(T_features, I_features)
        # T_features = CSFlow.l2_normalize_channelwise(T_features)
        # I_features = CSFlow.l2_normalize_channelwise(I_features)

        # work seperatly for each example in dim 1
        cosine_dist_l = []
        N = T_features.size()[0]
        for i in range(N):
            T_features_i = cs_flow.T_features_centered[i, :, :, :].unsqueeze(0)  # 1HWC --> 1CHW
            I_features_i = cs_flow.I_features_centered[i, :, :, :].unsqueeze(0).permute((0, 3, 1, 2))
            patches_PC11_i = cs_flow.patch_decomposition(T_features_i)  # 1HWC --> PC11, with P=H*W
            cosine_dist_i = torch.nn.functional.conv2d(I_features_i, patches_PC11_i)
            # cosine_dist_1HWC = cosine_dist_i.permute((0, 2, 3, 1))
            cosine_dist_l.append(cosine_dist_i.permute((0, 2, 3, 1)))  # back to 1HWC
            # cs_flow.cs_NHWC = cosine_dist_l[0] 
            # return cs_flow

        # cs_flow.cs_NHWC = cosine_dist_l[0] 
        # return cs_flow

        cs_flow.cosine_dist = torch.cat(cosine_dist_l, dim=0)

        cs_flow.raw_distances = - (cs_flow.cosine_dist - 1) / 2  ### why -

        relative_dist = cs_flow.calc_relative_distances()
        cs_flow.__calculate_CS(relative_dist)
        return cs_flow

    def calc_relative_distances(self, axis=TensorAxis.C):
        epsilon = 1e-5
        div = torch.min(self.raw_distances, dim=axis, keepdim=True)[0]
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    @staticmethod
    def sum_normalize(cs, axis=TensorAxis.C):
        reduce_sum = torch.sum(cs, dim=axis, keepdim=True)
        cs_normalize = torch.div(cs, reduce_sum)
        return cs_normalize

    
    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size
        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT = T_features.mean(0, keepdim=True).mean(1, keepdim=True).mean(2, keepdim=True)
        # self.varT = T_features.var(0, keepdim=True).var(1, keepdim=True).var(2, keepdim=True)
        self.T_features_centered = torch.sub(T_features, self.meanT)
        self.I_features_centered = torch.sub(I_features, self.meanT)

        # return self.T_features_centered, self.I_features_centered

    @staticmethod
    def l2_normalize_channelwise(features):
        norms = features.norm(p=2, dim=TensorAxis.C, keepdim=True)
        features = features.div(norms)
        return features

    @staticmethod
    def l2_normalize_channelwise_inplace(cs_flow):
        norms = cs_flow.T_features_centered.norm(p=2, dim=TensorAxis.C, keepdim=True)
        cs_flow.T_features_centered = cs_flow.T_features_centered.div(norms)
        norms = cs_flow.I_features_centered.norm(p=2, dim=TensorAxis.C, keepdim=True)
        cs_flow.I_features_centered = cs_flow.I_features_centered.div(norms)

    def patch_decomposition(self, T_features):
        # 1HWC --> 11PC --> PC11, with P=H*W
        (N, H, W, C) = T_features.shape
        P = H * W
        patches_PC11 = T_features.reshape(shape=(1, 1, P, C)).permute(dims=(2, 3, 0, 1))
        return patches_PC11

    @staticmethod
    def pdist2(x, keepdim=False):
        sx = x.shape
        x = x.reshape(shape=(sx[0], sx[1] * sx[2], sx[3]))
        differences = x.unsqueeze(2) - x.unsqueeze(1)
        distances = torch.sum(differences**2, -1)
        if keepdim:
            distances = distances.reshape(shape=(sx[0], sx[1], sx[2], sx[3]))
        return distances

    @staticmethod
    def calcR_static(sT, order='C', deformation_sigma=0.05):
        # oreder can be C or F (matlab order)
        pixel_count = sT[0] * sT[1]

        rangeRows = range(0, sT[1])
        rangeCols = range(0, sT[0])
        Js, Is = np.meshgrid(rangeRows, rangeCols)
        row_diff_from_first_row = Is
        col_diff_from_first_col = Js

        row_diff_from_first_row_3d_repeat = np.repeat(row_diff_from_first_row[:, :, np.newaxis], pixel_count, axis=2)
        col_diff_from_first_col_3d_repeat = np.repeat(col_diff_from_first_col[:, :, np.newaxis], pixel_count, axis=2)

        rowDiffs = -row_diff_from_first_row_3d_repeat + row_diff_from_first_row.flatten(order).reshape(1, 1, -1)
        colDiffs = -col_diff_from_first_col_3d_repeat + col_diff_from_first_col.flatten(order).reshape(1, 1, -1)
        R = rowDiffs ** 2 + colDiffs ** 2
        R = R.astype(np.float32)
        R = np.exp(-(R) / (2 * deformation_sigma ** 2))
        return R


class ContextualLoss(nn.Module):
    def __init__(self):
        super(ContextualLoss, self).__init__()

    def forward(self, I_features, T_features, deformation=False, dis=False):
        def from_pt2tf(Tpt):
            Ttf = Tpt.permute(0, 2, 3, 1)
            return Ttf
        # N x C x H x W --> N x H x W x C
        T_features_tf = from_pt2tf(T_features)
        I_features_tf = from_pt2tf(I_features)

        cs_flow = CSFlow.create_using_dotP(I_features_tf, T_features_tf, sigma=1.0)
        # return torch.mean(I_features - T_features)
        # cs_flow = CSFlow.create_using_L2(I_features_tf, T_features_tf, sigma=1.0)
        # sum_normalize:
        # To:
        cs = cs_flow.cs_NHWC

        if deformation:
            deforma_sigma = 0.001
            sT = T_features_tf.shape[1:2 + 1]
            R = CSFlow.calcR_static(sT, deformation_sigma=deforma_sigma)
            cs *= torch.Tensor(R).unsqueeze(dim=0).cuda()

        if dis:
            CS = []
            k_max_NC = torch.max(torch.max(cs, dim=1)[1], dim=1)[1]
            indices = k_max_NC.cpu()
            N, C = indices.shape
            for i in range(N):
                CS.append((C - len(torch.unique(indices[i, :]))) / C)
            score = torch.FloatTensor(CS)
        else:
            # reduce_max X and Y dims
            # cs = CSFlow.pdist2(cs,keepdim=True)
            k_max_NC = torch.max(torch.max(cs, dim=1)[0], dim=1)[0]
            # reduce mean over C dim
            CS = torch.mean(k_max_NC, dim=1)
            # score = 1/CS
            # score = torch.exp(-CS*10)
            score = -torch.log(CS)
        # reduce mean over N dim
        # CX_loss = torch.mean(CX_loss)
        score = score.mean()
        return score


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss

class CharbonnierMeanLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierMeanLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

def tv_loss(img, var):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    # h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    # loss = tv_weight * (h_variance + w_variance)
    # return loss

    w_variance = torch.sum((1 - var)[:,:,:,:-1] * (torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2)))
    h_variance = torch.sum((1 - var)[:,:,:-1,:] * (torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2)))
    loss = h_variance + w_variance
    return loss