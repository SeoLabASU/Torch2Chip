"""
Transformer layer fuser
"""
import copy
import torch
import torch.nn as nn
from methods.base import MulShift
from models import TransformerEncoder
from methods import LinearMulShift, LinearMulShiftReLU, QBaseLinear, MulQuant

class XformerFuser(object):
    def __init__(self, model:nn.Module):
        self.model = model
        
        # flag
        self.flag = False
    
    def inference(self, model:nn.Module):
        """
        Switch to inference mode
        """
        for n, m in model.named_modules():
            if "msa" in n:
                if hasattr(m, "inference"):
                    if not ('qq' in n or 'kq' in n or 'vq' in n or 'oq' in n):
                        m.inference()

    def layers(self):
        pass

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
                setattr(msa, "vdeq", MulQuant(nbit=msa.wbit))

                # insert back
                setattr(m, "msa", msa)

        return fused_model

    def mlp_params(self, mlp:nn.Module):
        xscales = []
        for m in mlp.modules():
            if isinstance(m, QBaseLinear):
                if hasattr(m.aq, "scale"):
                    xscales.append(m.aq.scale.data)
                else:
                    xscales.append(torch.tensor(1.0))
        return xscales

    def mlp_fuser(self, model:nn.Module):
        fused_model = copy.deepcopy(model) 
        for n, m in fused_model.named_modules():
            if isinstance(m, TransformerEncoder):
                mlp = m.mlp

                # get scalers
                qscales = self.mlp_params(mlp)
                counter = 0

                fmlp = []
                for k in mlp.modules():
                    if isinstance(k, QBaseLinear):
                        qflag = True if counter==0 else False
                        
                        # merged module holder
                        tmp = LinearMulShiftReLU(k.in_features, k.out_features, k.wbit, k.abit, qflag=qflag, obit=k.wbit)

                        # scalers
                        if counter == 0:
                            sin = 1.0
                            sout = qscales[counter+1]
                            
                            # switch mode
                            # k.aq.inference()                        
                            k.wq.inference()
                            tmp.scaler.nlv = 2**k.wbit - 1
                        else:
                            sin = k.aq.scale.data
                            sout = 1.0
                            
                            # switch mode
                            k.aq.inference()                        
                            k.wq.inference()
                            
                            setattr(k, "aq", nn.Identity())

                        sw = k.wq.scale.data

                        # update scaling
                        tmp.scaler.scale = sout / (sw * sin)

                        # update module
                        setattr(tmp, "linear", k)
                        tmp.linear.bias.data.mul_(sw * sin)

                        fmlp.append(tmp)
                        counter += 1

                fmlp = nn.Sequential(*fmlp)
                setattr(m, "mlp", fmlp)

                # # layernorm fuser
                # norm1 = MulShift()
                # norm2 = MulShift()

                # std1 = torch.sqrt(m.la1.running_var.data + m.la1.eps)
                # std2 = torch.sqrt(m.la2.running_var.data + m.la2.eps)
    
                # # scaler and bias
                # norm1.scale.data = m.la1.weight.div(std1)
                # norm2.scale.data = m.la2.weight.div(std2)

                # norm1.bias.data = m.la1.bias - m.la1.weight.mul(m.la1.running_mean.data).div(std1)
                # norm2.bias.data = m.la2.bias - m.la2.weight.mul(m.la2.running_mean.data).div(std2)

                # # replace the module
                # setattr(m, "la1", norm1)
                # setattr(m, "la2", norm2)

        return fused_model

    def fuse(self):
        model = self.encoder_fuser()
        self.inference(model)
        
        fused_model = self.mlp_fuser(model)
        return fused_model
        