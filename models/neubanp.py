from abc import ABC

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.exponential import Exponential as Exp
from attrdict import AttrDict

from models.modules import CrossAttnEncoder, NeuCrossAttnEncoder, NeuBootsEncoder, Decoder


class NEUBANP(nn.Module):
    """
    structure: NeuBoots Encoder 2개 평행하게 -> concatenation -> NeuBoots Decoder
    predict: 추론 시 이용. xc, yc, xt를 받아 yt에 대한 여러개의 예측(ys, mean, std)과 bootstrapping weight w를 리턴
    forward: 학습 시 이용. batch를 받아 batch 안의 모든 (xc,yc,xt)에 대해 predict를 수행하고, output과 yt를 비교하여 loss 리턴
    """

    def __init__(self,
                 dim_x=1,
                 dim_y=1,
                 dim_hid=128,
                 enc_v_depth=4,
                 enc_qk_depth=2,
                 enc_pre_depth=4,
                 enc_post_depth=2,
                 dec_depth=3,
                 yenc=True,
                 wenc=True,
                 wagg=True,
                 wloss=True,
                 l2=False,
                 wattn=False):
        super(NEUBANP, self).__init__()

        self.yenc = yenc
        self.wenc = wenc
        self.wloss = wloss
        self.l2 = l2
        self.wattn = wattn

        if self.wattn:
            self.enc1 = NeuCrossAttnEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                v_depth=enc_v_depth,
                qk_depth=enc_qk_depth)
        else:
            self.enc1 = CrossAttnEncoder(
                dim_x=dim_x,
                dim_y=dim_y,
                dim_hid=dim_hid,
                v_depth=enc_v_depth,
                qk_depth=enc_qk_depth)

        self.enc2 = NeuBootsEncoder(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_hid=dim_hid,
            self_attn=True,
            pre_depth=enc_pre_depth,
            post_depth=enc_post_depth,
            yenc=yenc,
            wenc=wenc,
            wagg=wagg)

        self.dec = Decoder(
            dim_x=dim_x,
            dim_y=dim_y,
            dim_enc=2 * dim_hid,
            dim_hid=dim_hid,
            depth=dec_depth,
            neuboots=True)

    def predict(self, xc, yc, xt, num_samples=1, num_bs=50):
        # botorch 사용하기 위해 추가된 statement
        if xc.shape[-3] != xt.shape[-3]:
            xt = xt.transpose(-3, -2)

        """
        주어진 데이터(xc,yc,xt)에 대한 예측 값(ys, mean, std)과 bootstrapping weight w를 리턴
        - 같은 데이터(xc,yc,xt)에 대해 여러번(num_samples)을 추론함
        - 매 추론 시 여러개(num_bs)의 bootstrapping weight를 이용하여 여러 개의 y와 그 mean, std을 w와 함께 리턴
        xc: context feature [B,Nc,Dx]
        yc: context label [B,Nc,Dy]
        xt: target feature [B,Nt,Dx]
        """
        Ns, B, Nbs, Nc, Nt = num_samples, xc.size(0), num_bs, xc.size(-2), xt.size(-2)
        device = xc.device

        xc_bs = torch.stack([torch.stack([xc] * Nbs, -3)] * Ns, 0).squeeze(0)  # [Ns,B,Nbs,Nc,Dx]
        yc_bs = torch.stack([torch.stack([yc] * Nbs, -3)] * Ns, 0).squeeze(0)  # [Ns,B,Nbs,Nc,Dy]
        xt_bs = torch.stack([torch.stack([xt] * Nbs, -3)] * Ns, 0).squeeze(0)  # [Ns,B,Nbs,Nt,Dx]

        wc_bs = Exp(torch.ones(Ns, B, Nbs, Nc, 1)).sample().squeeze(0).to(device)  # [Ns,B,Nbs,Nc,1]

        if not self.yenc:
            yc_bs = torch.tensor([]).to(device)  # yenc를 안 쓰면 빈 텐서로 만들어서 나중에 concat 시 없어지도록 함.

        if self.wattn:
            enc1 = self.enc1(xc_bs, yc_bs, xt_bs, wc_bs)  # [Ns,B,Nbs,Nt,Eh]
        else:
            enc1 = self.enc1(xc_bs, yc_bs, xt_bs)  # [Ns,B,Nbs,Nt,Eh]
        enc2 = self.enc2(xc_bs, yc_bs, wc_bs)  # [Ns,B,Nbs,Eh]
        enc2 = torch.stack([enc2] * Nt, -2)  # [Ns,B,Nbs,Nt,Eh]
        encoded = torch.cat([enc1, enc2], -1)  # [Ns,B,Nbs,Nt,2Eh]
        out = self.dec(encoded, xt_bs)  # [Ns,B,Nbs,Nt,Dy]

        outs = AttrDict()
        outs.mean = out.mean(dim=-3)  # [Ns,B,Nt,Dy], dim=-3: dimension of Nbs
        outs.std = out.std(dim=-3)  # [Ns,B,Nt,Dy]
        outs.ys = out  # [Ns,B,Nbs,Nt,Dy]
        outs.ws = wc_bs  # [Ns,B,Nbs,Nc,1]  나중에 wagg와 wloss에서 사용할 수 있도록 bootstrapping weight 값들도 리턴

        return outs
        # {"mean": [Ns,B,Nt,Dy], "std": [Ns,B,Nt,Dy], "ys": [Ns,B,Nbs,Nt,Dy], "ws": [Ns,B,Nbs,Nc,1]}

    def forward(self, batch, num_samples=1, num_bs=50, loss="nll", alpha=0.5, beta=0.5, eps=1e-3, reduce_ll=True):
        """
        predict를 불러 예측하고, loss를 리턴
        batch.xc: [B,Nc,Dx]
        batch.yc: [B,Nc,Dy]
        batch.xt: [B,Nt,Dx]
        batch.yt: [B,Nt,Dy]
        batch.x: [B,N,Dx]
        batch.y: [B,N,Dy]
        device: torch.device("cpu") or torch.device("cuda")
        reduce_ll: loss 값을 각 batch 각 data point마다 남겨둘지, 전부 mean 할지
        num_samples: 주어진 batch에 대해 수행할 prediction 갯수
        """
        outs = AttrDict()

        pred = self.predict(batch.xc, batch.yc, batch.x, num_samples, num_bs)
        y_hat, mu_hat, sigma_hat, ws = pred.ys, pred.mean, pred.std, pred.ws.squeeze(-1)
        # y_hat: [Ns,B,Nbs,N,Dy]
        # mu_hat: [Ns,B,N,Dy]
        # sigma_hat: [Ns,B,N,Dy]
        # ws: [Ns,B,Nbs,Nc]

        y = batch.y  # [B,N,Dy]
        Nc = batch.xc.size(1)

        mu_hat_c = mu_hat[..., :Nc, :]  # [Ns,B,Nc,Dy]
        sigma_hat_c = sigma_hat[..., :Nc, :]  # [Ns,B,Nc,Dy]

        mu_hat_t = mu_hat[..., Nc:, :]  # [Ns,B,Nt,Dy]
        sigma_hat_t = sigma_hat[..., Nc:, :]  # [Ns,B,Nct,Dy]

        y_c = y[..., :Nc, :]  # [B,Nc,Dy]
        y_hat_c = y_hat[..., :Nc, :]  # [Ns,B,Nbs,Nc,Dy]

        y_t = y[..., Nc:, :]  # [B,Nt,Dy]
        y_hat_t = y_hat[..., Nc:, :]  # [Ns,B,Nbs,Nt,Dy]

        """
        context: NLL or L2 (if self.l2 = True)
        target: L2
        """
        ctx_l2 = compute_l2(y_hat_c, y_c, num_samples=num_samples)  # [Ns,B,Nc]
        tar_l2 = compute_l2(y_hat_t, y_t, num_samples=num_samples)  # [Ns,B,Nt]

        if loss == "betanll":
            ctx_nll, ctx_nll_mu, ctx_nll_sigma = compute_beta_nll(mu=mu_hat_c, sigma=sigma_hat_c,
                                                                  y=y_c, ws=ws, num_samples=num_samples,
                                                                  beta=beta, eps=eps)  # [Ns,B,Nc]
            tar_nll, tar_nll_mu, tar_nll_sigma = compute_beta_nll(mu=mu_hat_t, sigma=sigma_hat_t,
                                                                  y=y_t, ws=None, num_samples=num_samples,
                                                                  beta=beta, eps=eps)  # [Ns,B,Nc]
            ctx_loss = ctx_nll  # [Ns,B,Nc]
            tar_loss = tar_nll

        elif loss == "nll":
            ctx_nll = compute_nll(mu=mu_hat_c, sigma=sigma_hat_c, y=y_c,
                                  num_samples=num_samples, ws=ws, eps=eps)
            tar_nll = compute_nll(mu=mu_hat_t, sigma=sigma_hat_t, y=y_t,
                                  num_samples=num_samples, ws=None, eps=eps)
            ctx_loss = ctx_nll  # [Ns,B,Nc]
            tar_loss = tar_nll  # [Ns,B,Nt]

        elif loss == "l2":
            ctx_loss = ctx_l2  # [Ns,B,Nc]
            tar_loss = tar_l2

        else:
            raise NotImplementedError

        _ctx_loss = (1 - alpha) * ctx_loss  # [Ns,B,Nc]
        _tar_loss = alpha * tar_loss  # [Ns,B,Nt]

        if reduce_ll:
            ctx_loss = ctx_loss.mean()  # [1,]
            tar_loss = tar_loss.mean()  # [1,]
            loss = _ctx_loss.mean() + _tar_loss.mean()  # [1,]
        else:
            loss = torch.cat([_ctx_loss, _tar_loss], dim=-1)  # [Ns,B,N]

        """
        원하는 부분 켜서 logging
        """
        # outs.ctx_nll_mu = ctx_nll_mu.mean()
        # outs.ctx_nll_sigma = ctx_nll_sigma.mean()
        # outs.tar_nll_mu = tar_nll_mu.mean()
        # outs.tar_nll_sigma = tar_nll_sigma.mean()
        # outs.nll_mu = ctx_nll_mu.mean() * alpha + tar_nll_mu.mean() * (1 - alpha)
        # outs.nll_sigma = ctx_nll_sigma.mean() * alpha + tar_nll_sigma.mean() * (1 - alpha)
        # outs.ctx_l2 = ctx_l2.mean()
        # outs.tar_l2 = tar_l2.mean()
        # outs.l2 = ctx_l2.mean() * alpha + tar_l2.mean() * (1 - alpha)
        outs.ctx_loss = ctx_loss
        outs.tar_loss = tar_loss
        outs.loss = loss
        return outs  # {"ctx_nll_mu", "ctx_nll_sigma", "ctx_l2", "tar_l2", "ctx_loss", "tar_loss", loss"}


def compute_nll(mu, sigma, y, ws=None, num_samples=1, eps=1e-3):
    Ns = num_samples
    py = Normal(mu, sigma + eps)  # [Ns,B,N,Dy]
    y = torch.stack([y] * Ns, 0).squeeze(0)  # [Ns,B,N,Dy]
    ll = py.log_prob(y).sum(-1)  # [Ns,B,N]

    if ws is not None:
        Nbs = ws.size(-2)
        ll = torch.stack([ll] * Nbs, -2)  # [Ns,B,Nbs,N]
        ll = (ll * ws).mean(-2)  # [Ns,B,N]

    return - ll  # [Ns,B,N]


# mu,sigma : [Ns,B,N,Dy], y: [B,N,Dy] ws: [Ns,B,Nbs,N]
def compute_beta_nll(mu, sigma, y, ws=None, num_samples=1, beta=0.5, eps=1e-3):
    Ns = num_samples
    sigma = sigma + eps
    y = torch.stack([y] * Ns, dim=0)  # [Ns,B,N,Dy]

    ll_mu = - (((y - mu) ** 2) / (2 * sigma ** 2)).sum(-1)  # [Ns,B,N]
    ll_sigma = - torch.log(sigma).sum(-1)  # [Ns,B,N]

    if ws is not None:  # [Ns,B,Nbs,N]
        Nbs = ws.size(-2)
        _ll_mu = torch.stack([ll_mu] * Nbs, -2)  # [Ns,B,Nbs,N]
        _ll_sigma = torch.stack([ll_sigma] * Nbs, -2)  # [Ns,B,Nbs,N]

        _ll_mu = (_ll_mu * ws).mean(-2)  # [Ns,B,N]
        _ll_sigma = (_ll_sigma * ws).mean(-2)  # [Ns,B,N]

        ll = 2 * beta * _ll_mu + (2 - 2 * beta) * _ll_sigma  # [Ns,B,N]

    else:
        ll = 2 * beta * ll_mu + (2 - 2 * beta) * ll_sigma  # [Ns,B,N]

    return - ll, - ll_mu, - ll_sigma  # [Ns,B,N] all


def compute_l2(y_hat, y, num_samples=1):  # pred: [Ns,B,Nbs,N,Dy], y: [B,N,Dy]
    Ns = num_samples
    Nbs = y_hat.size(-3)
    y = torch.stack([torch.stack([y] * Nbs, -3)] * Ns, 0).squeeze(0)  # [Ns,B,Nbs,N,Dy]

    l2 = ((y_hat - y) ** 2).sum(-1).mean(-2)  # [Ns,B,N]
    return l2  # [Ns,B,N]
