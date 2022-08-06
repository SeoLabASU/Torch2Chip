"""
Transformer layer fuser
"""
import copy
import torch
import torch.nn as nn
from models import TransformerEncoder
from methods import LinearMulShift, QBaseLinear

class XformerFuser(object):
    def __init__(self, model:nn.Module):
        self.model = model
        
        # flag
        self.flag = False

        # layers
        self.groups = []

        # parameters
        self.xscales = {}
        self.xbound = {}
    
    def inference(self, model:nn.Module):
        """
        Switch to inference mode
        """
        for n, m in model.named_modules():
            if "msa" in n:
                if hasattr(m, "inference"):
                    if not ('qq' in n or 'kq' in n or 'vq' in n or 'oq' in n):
                        m.inference()

    def fused_linear(self, linear:QBaseLinear, qflag:bool=True, obit:int=32):
        f = LinearMulShift(linear.in_features, linear.out_features, 
                linear.wbit, linear.abit, linear.train_flag, qflag=qflag, obit=obit)
        setattr(f, "linear", linear)
        return f

    def encoder_fuser(self):
        fused_model = copy.deepcopy(self.model) 
        for n, m in fused_model.named_modules():
            if isinstance(m, TransformerEncoder):
                # multi head attention
                msa = m.msa

                # fused linear layers
                fq = self.fused_linear(msa.q, obit=msa.qq.nbit)
                fk = self.fused_linear(msa.k, obit=msa.kq.nbit)
                fv = self.fused_linear(msa.v, obit=msa.vq.nbit)
                fo = self.fused_linear(msa.o, qflag=False)

                # fused q modules
                sq = 1 / (msa.q.wq.scale.data*msa.aq.scale.data)
                sk = 1 / (msa.k.wq.scale.data*msa.aq.scale.data)
                sv = 1 / (msa.v.wq.scale.data*msa.aq.scale.data)
                so = 1 / (msa.o.wq.scale.data*msa.oq.scale.data)

                # update scalers
                fq.scaler.scale = sq * msa.qq.scale
                fk.scaler.scale = sk * msa.kq.scale
                fv.scaler.scale = sv * msa.vq.scale
                fo.scaler.scale = so

                # update bias
                fq.linear.bias.data.div_(sq)
                fk.linear.bias.data.div_(sk)
                fv.linear.bias.data.div_(sv)
                fo.linear.bias.data.div_(so)

                # update dequantizer scaler
                msa.deq.scale = 1 / (msa.qq.scale * msa.kq.scale * msa.sqrt_d)
                msa.vdeq.scale = (1 / msa.vq.scale) * msa.oq.scale

                # replace the original module
                setattr(msa, "q", fq)
                setattr(msa, "k", fk)
                setattr(msa, "v", fv)
                setattr(msa, "o", fo)

                # integer only multiplication of o
                msa.o.linear.aq = nn.Identity()

                # delete the original quant module
                setattr(msa, "qq", nn.Identity())
                setattr(msa, "kq", nn.Identity())
                setattr(msa, "vq", nn.Identity())
                setattr(msa, "oq", nn.Identity())

                # insert back
                setattr(m, "msa", msa)

        return fused_model

    