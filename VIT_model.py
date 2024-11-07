import torch 
from torch import nn
from tf_VIT import TransformerEncoder

class VIT(nn.Module):
    def __init__(self):
        super().__init__()
        self._emb_layer=nn.Conv2d(1,64,14,14,bias=False)
        self._tf_layer=TransformerEncoder(
            num_layers=2,
            input_dim=64,
            hide_dim=48,
            n_q_heads=2,
            n_kv_heads=1,
            max_len=5
        )
        self._out_layer=nn.Linear(64,10,bias=False)

        _cls_token=torch.randn(64)
        self.register_buffer("_cls_token",_cls_token)
    def forward(self,x):
        _bn,_ch,_h,_w=x.shape
        
if __name__=="__main__":
    vit=VIT()



        