# -*- coding: utf-8 -*-
"""
@inproceedings{wang2023focus,
title={Focus Your Attention when Few-Shot Classification},
author={Haoqing Wang and Shibo Jie and Zhi-Hong Deng},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=uFlE0qgtRO}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from qpth.qp import QPFunction
import numpy as np

from core.model.finetuning.finetuning_model import FinetuningModel

class FORTModel(FinetuningModel):
    def __init__(self, encoder, n_way, **kwargs):
        super(FinetuningModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.n_way = n_way
        self.head = nn.Linear(encoder.feat_dim, n_way)

        self.hooks = []
        for name, module in self.named_modules():
            if 'attn_drop' in name:
                self.hooks.append(module.register_forward_hook(self.get_attention))
            if 'blocks.11.norm1' in name:
                self.hooks.append(module.register_backward_hook(self.get_gradient))
        self.attentions = []
        self.gradients = []

    def set_forward(self, img):
        x, attn = self.encoder(img)
        scores = self.head(x)
        return scores, attn.mean(1)

    def set_forward_loss(self, x, labels):
        logits, _ = self.set_forward(x)
        loss = - (labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        return loss

    def get_attention(self, output):
        self.attentions.append(output.cpu())

    def get_gradient(self, grad_input):
        self.gradients.append(grad_input[0].cpu())

    def reset_protonet_head(self, xs):
        self.encoder.eval()
        norm = 4000
        with torch.no_grad():
            feat, _ = self.encoder(xs)
        z_p = feat.reshape(self.n_way, -1, feat.size(-1)).mean(1)
        state_dict = dict(weight=2.*z_p/norm, bias=-(z_p*z_p).sum(1)/norm)
        self.head.load_state_dict(state_dict)

    def reset_r2d2_head(self, xs, temp=20., lam=50):
        self.encoder.eval()
        with torch.no_grad():
            support, _ = self.encoder(xs)
        num_support, d = support.size()
        support_labels = torch.from_numpy(np.repeat(range(self.n_way), num_support//self.n_way)).cuda()  # (num_support)
        support_labels_one_hot = one_hot(support_labels, self.n_way)  # (num_support, n_way)

        ridge_sol = torch.mm(support, support.transpose(0, 1)) + lam * torch.eye(num_support).cuda()  # (num_support, num_support)
        ridge_sol, _ = torch.solve(torch.eye(num_support).cuda(), ridge_sol)
        ridge_sol = torch.mm(support.transpose(0, 1), ridge_sol)  # (d, num_support)

        weight = torch.mm(ridge_sol, support_labels_one_hot).transpose(0, 1)  # (n_way, d)
        state_dict = dict(weight=weight*temp, bias=torch.zeros(self.n_way).cuda())
        self.head.load_state_dict(state_dict)

    def reset_meta_optnet_head(self, xs, temp=20., C_reg=0.1, maxIter=15):
        self.encoder.eval()
        with torch.no_grad():
            support, _ = self.encoder(xs)
        num_support, d = support.size()
        support_labels = torch.from_numpy(np.repeat(range(self.n_way), num_support//self.n_way)).cuda()

        kernel_matrix = torch.mm(support, support.transpose(0, 1))
        id_matrix = torch.eye(self.n_way).cuda()
        block_kernel_matrix = kronecker(kernel_matrix, id_matrix)
        block_kernel_matrix += 1.0 * torch.eye(self.n_way * num_support).cuda()
        support_labels_one_hot = one_hot(support_labels, self.n_way)

        G = block_kernel_matrix
        e = -1. * support_labels_one_hot.flatten()
        C = torch.eye(self.n_way * num_support).cuda()
        h = C_reg * support_labels_one_hot.flatten()
        A = kronecker(torch.eye(num_support).cuda(), torch.ones(1, self.n_way).cuda())
        b = torch.zeros(num_support).cuda()
        qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())
        qp_sol = qp_sol.reshape(num_support, self.n_way)

        weight = qp_sol.transpose(0, 1) @ support
        state_dict = dict(weight=weight*temp, bias=torch.zeros(self.n_way).cuda())
        self.head.load_state_dict(state_dict)

def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()  # (n_batch, m, depth) or (m, depth)
    index = indices.reshape(indices.size()+torch.Size([1]))  # (n_batch, m, 1) or (m, 1)
    if len(indices.size()) < 2:
        encoded_indicies = encoded_indicies.scatter_(1, index, 1)
    else:
        encoded_indicies = encoded_indicies.scatter_(2, index, 1)
    return encoded_indicies

def kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(-1)
    matrix2_flatten = matrix2.reshape(-1)
    return torch.mm(matrix1_flatten.unsqueeze(1), matrix2_flatten.unsqueeze(0)).reshape(list(matrix1.size())+list(matrix2.size())).permute([0, 2, 1, 3]).reshape(matrix1.size(0)*matrix2.size(0), matrix1.size(1)*matrix2.size(1))

def get_first_comp(inp):
    inp[torch.isnan(inp)] = 0.
    inp = inp - inp.mean(1, keepdim=True)
    U, S, V = torch.svd(inp, some=False)
    projection = inp @ V[:, :, :1]
    return projection.squeeze()

def importance(net, inp, lab, lamb=1.):
    net.zero_grad()
    output, _ = net(inp)
    category_mask = torch.zeros(output.size()).to(output.device)
    category_mask = category_mask.scatter_(1, lab.unsqueeze(1), 1)
    logit = (output * category_mask).sum(-1).mean()
    logit.backward()
    net.zero_grad()
    attns, grads = net.attentions, net.gradients

    grad = get_first_comp(grads[0][:, 1:].cpu())

    with torch.no_grad():
        result = torch.eye(attns[0].size(-1) - 1).unsqueeze(0).to(attns[0].device)  # (1, L, L)
        for attn in attns:
            attn_fused = attn.min(1)[0][:, 1:, 1:] + lamb * grad.unsqueeze(1)
            _, indices = attn_fused.topk(int(attn_fused.size(-1) * 0.9), -1, False)
            attn_fused = attn_fused.scatter_(-1, indices, 0)

            I = torch.eye(attn_fused.size(-1)).unsqueeze(0).to(attn_fused.device)
            a = (attn_fused + I) / 2.
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result
    imp = result.mean(1)

    del net.attentions, net.gradients
    for hook in net.hooks:
        hook.remove()

    return imp.cuda()

def imp_to_focus(imp, P):
    _, ids_shuffle = torch.sort(imp, descending=True, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    focus = torch.ones_like(imp)
    focus[:, P:] = 0
    focus = torch.gather(focus, dim=1, index=ids_restore)

    focus = imp * focus
    focus = focus * P / focus.sum(-1, keepdim=True)
    return focus

def ce_loss(y_pred, y_true):
    return - (y_true * F.log_softmax(y_pred, dim=-1)).sum(dim=-1).mean()

def finetune_model(model, xs, ys, focus, params, n_way=5, n_support=5):
    model.train()  # 设置模型为训练模式
    batch_size = n_way  # 定义每批次的大小
    support_size = n_way * n_support  # 支持集大小
    loss_fn = nn.CrossEntropyLoss().cuda()  # 使用交叉熵损失

    # 选择需要微调的参数
    parameters = []
    num = 0
    for n, p in model.named_parameters():
        if ('ssf_' in n) or ('head' in n):
            parameters.append(p)
            num += p.numel()

    # 为微调的参数设置优化器
    opt = torch.optim.AdamW(parameters, lr=params.ft_lr)

    for epoch in range(params.ft_epoch):
        rand_id = np.random.permutation(support_size)
        for j in range(0, support_size, batch_size):
            opt.zero_grad()
            selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size)]).cuda()
            x_batch = xs[selected_id]  # 获取当前批次的输入
            y_batch = ys[selected_id]  # 获取当前批次的标签
            focus_batch = focus[selected_id]  # 获取当前批次的关注区域

            scores, attn = model(x_batch)  # 前向传播得到输出和注意力
            focus_batch = focus_batch.unsqueeze(1).expand_as(attn).reshape(-1, attn.size(-1))
            attn = attn.reshape(-1, attn.size(-1)) / params.tau  # 温度调节
            loss = loss_fn(scores, y_batch) + params.alpha * ce_loss(attn, focus_batch)  # 计算微调损失
            loss.backward()  # 反向传播
            opt.step()  # 更新模型参数

    del opt  # 删除优化器
    torch.cuda.empty_cache()  # 清理GPU内存
