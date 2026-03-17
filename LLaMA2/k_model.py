import math

import self
import torch
import torch.nn.functional as F
from scipy.signal import max_len_seq
from torch import nn
from typing import Any, Optional, Tuple
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-k"
    def __init__(
            self,
            dim:int = 768, # 模型维度
            n_layers:int = 12, # Transformer层数
            n_heads:int = 16, # 注意力机制的头数
            n_kv_heads:int = 8, # 键值头的数量
            vocab_size:int = 6144, # 词汇表大小
            hidden_dim:int = None, # 隐藏层维度
            multiple_of:int = 64,
            norm_eps:float = 1e-5, # 归一化层的eps
            max_seq_len:int = 512, # 最大序列长度
            dropout:float = 0.0, # dropout概率
            flash_attn:bool = True, # 是否使用Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps:float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        # forward函数是模型的前向传播
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def repeat_kv(x:torch.Tensor, n_rep:int) -> torch.Tensor:
    # 获取输入张量的大小：批次大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, slen, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x

    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:,:,:,None,:]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# 构造获得旋转嵌入的实部和虚部的函数
def precompute_freqs_cis(dim:int,end:int,theta:float=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)]).float() / dim)
    t=torch.arange(end,device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos= torch.cos(freqs)
    freqs_sin= torch.sin(freqs)
    return freqs_cos, freqs_sin

# 调整 freqs_cis 的形状，使其在进行广播操作时与 x 的维度对齐，从而能够进行正确的张量运算。
def reshape_for_broadcast(freqs_cis:torch.Tensor,x:torch.Tensor):
    ndim = x.ndim

    assert 0 <= 1 < ndim

    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(shape)

def apply_rotary_emb(
        xq:torch.Tensor,
        xk:torch.Tensor,
        freqs_cos:torch.Tensor,
        freqs_sin:torch.Tensor,
)->tuple[torch.Tensor,torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i=xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i=xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_i)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 还原成原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq),xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self,args=ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小
        model_parallel_size = 1
        # 本地计算头的数量
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头的数量
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout
        self.dropout = args.dropout

        # 检查是否需要使用Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 手动实现注意力机制，创建一个上三角矩阵，遮掩未来信息
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len),float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer('mask', mask)

    def forward(self,x:torch.Tensor,freqs_cos:torch.Tensor,freqs_sin:torch.Tensor):
        # 获取批次大小和序列长度
        bsz, seqlen, _ = x.shape

        xq,xk,xv = self.wq(x),self.wk(x),self.wv(x)
        # 调整形状以适应头的维度
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入
        xq, xk = apply_rotary_emb(xq,xk,freqs_cos,freqs_sin)

        xk = repeat_kv(xk,self.n_rep)
        xv = repeat_kv(xv,self.n_rep)

        xq = xq.transpose(1,2)
        xk = xk.transpose(1,2)
        xv = xv.transpose(1,2)

        # 根据是否支持Flash Attention，选择实现方式
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq,xk,xv,attn_mask=None,dropout_p=self.dropout if self.training else 0.0,is_causal=True)
        else:
            scores = torch.matmul(xq,xk.transpose(2,3))/math.sqrt(self.head_dim)
            assert hasattr(self,'mask')
            scores = scores + self.mask[:,:,:seqlen,:seqlen]
            scores = F.softmax(scores.float(),dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores,xv)

        # 恢复时间维度并合并头
        output = output.transpose(1,2).contiguous().view(bsz, seqlen, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class MLP(nn.Module):
    def __init__(self,dim:int,hidden_dim:int,multiple_of:int,dropout:float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of -1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class DecoderLayer(nn.Module):
    def __init__(self, layer_id:int, args=ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        # 定义LLaMA2Attention对象，用于进行多头注意力计算
        self.attention = Attention(args)
        # 定义LLaMAMLP对象，用于进行前馈神经网络计算
        self.feed_forward = MLP(
            dim = args.dim,
            hidden_dim = args.hidden_dim,
            multiple_of = args.multiple_of,
            dropout = args.dropout
        )
        # 定义层的ID
        self.layer_id = layer_id
        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self,x,freq_cos,freq_sin):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x),freq_cos,freq_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(PreTrainedModel):
    config_class = ModelConfig
    last_loss: Optional[torch.Tensor]
    def __init__(self,args:ModelConfig=None):
        super().__init__(args)
        # 初始化模型参数
        self.args=args
        # 词汇表大小
        self.vocab_size = args.vocab_size
        # 层数
        self.n_layers = args.n_layers

        # 词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout层
        self.drop = nn.Dropout(args.dropout)
        # Decoder层
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # 归一化层
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 输出层
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 共享词嵌入层和输出层的权重
        self.tok_embeddings.weight = self.output.weight
        # 预计算相对位置嵌入的频率
        freq_cos, freq_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freq_cos", freq_cos, persistent=False)
        self.register_buffer("freq_sin", freq_sin, persistent=False)

        # 初始化所有权重
        self.apply(self._init_weights)
        # 对残差投影进行特殊的缩放初始化
        for pn,p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('w0.weight'):
                torch.nn.init.normal_(p, mean=0, std=0.02 / math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self._no_split_modules = [name for name, _ in self.named_parameters()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, tokens:torch.Tensor, targets:Optional[torch.Tensor]=None, **kwargs):
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']

        # 前向传播函数
        _bsz, seqlen = tokens.shape
        # 通过词嵌入层和Dropout层
        h = self.tok_embeddings(tokens)
        h = self.drop(h)
        # 获得相对位置嵌入的频率
        freq_cos = self.freq_cos[:seqlen]
        freq_sin = self.freq_sin[:seqlen]

        # 通过Decoder层
        for layer in self.layers:
            h = layer(h,freq_cos,freq_sin)
        # 通过归一化层
        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            logits = self.output(h[:,[-1],:])
            self.last_loss = None

        # 设置输出
        return CausalLMOutputWithPast(logits=logits, loss=self.last_loss)

    @torch.inference_mode()
    def generate(self, idx, stop_id=None,max_new_tokens=256,temperature=1.0,top_k=None):
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:,-self.args.max_seq_len:]

            # 前向传播获取序列中最后一个位置的 logits
            logits = self(idx_cond).logits
            logits = logits[:,-1,:]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits,k=1)
            else:
                # 缩放logits并应用softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits,k=min(top_k,logits.size(-1)))
                    logits[logits<v[:,[-1]]] = -float('Inf')
                probs = F.softmax(logits,dim=-1)
                idx_next = torch.multinomial(probs,1)

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx,idx_next),dim=1)

        return idx[:,index:]
