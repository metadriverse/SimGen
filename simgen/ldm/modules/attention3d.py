from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from simgen.ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


# modified following GenAD-private-genad_improved_yjz/sgm/modules/attention3d.py
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., zero_init=False):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

        if zero_init:
            nn.init.constant_(self.net[-1].weight, 0)
            nn.init.constant_(self.net[-1].bias, 0)


    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class RelativePosition(nn.Module):
    """
    https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
    """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings

# modified following GenAD-private-genad_improved_yjz/sgm/modules/attention3d.py
class TemporalAttention(nn.Module):
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=0.,
            use_relative_position=False,  # use relative positional representation in temporal attention or not
            temporal_length=None,  # for relative positional representation
            zero_init=False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.context_dim = context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.temporal_length = temporal_length
        self.use_relative_position = use_relative_position
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        if use_relative_position:
            assert temporal_length is not None
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)

        if zero_init:
            nn.init.constant_(self.to_out[0].weight, 0)
            nn.init.constant_(self.to_out[0].bias, 0)

    def _forward(self, x, mask=None):
        num_heads = self.heads

        # calculate qkv
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=num_heads), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
        # relative positional embedding
        if self.use_relative_position:
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum("b t d, t s d -> b t s", q, k2) * self.scale
            sim += sim2

        # mask attention
        if mask is not None:
            mask = mask.to(sim)
            max_neg_value = -1e9
            sim = sim + (1 - mask) * max_neg_value  # 1: perform attention, 0: no attention

        # attend to values
        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)

        # relative positional embedding
        if self.use_relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum("b t s, t s d -> b t d", attn, v2)
            out += out2

        # merge attention heads
        out = rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)
        return self.to_out(out)

    def forward(self, x, mask=None):
        forward_inputs = (x, )
        return checkpoint(self._forward, forward_inputs, self.parameters(), True)

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

# modified following GenAD-private-genad_improved_yjz/sgm/modules/attention3d.py
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., zero_init=False, checkpoint=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        if zero_init:
            nn.init.constant_(self.to_out[0].weight, 0)
            nn.init.constant_(self.to_out[0].bias, 0)
        self.checkpoint = checkpoint

    def _forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)
    def forward(self, x, context=None, mask=None):
        forward_inputs = (x, context)
        return checkpoint(self._forward, forward_inputs, self.parameters(), self.checkpoint)

# modified following GenAD-private-genad_improved_yjz/sgm/modules/attention3d.py
class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, zero_init=False, checkpoint=True):
        super().__init__()
        # print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
        #       f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

        if zero_init:
            nn.init.constant_(self.to_out[0].weight, 0)
            nn.init.constant_(self.to_out[0].bias, 0)

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

### start GenAD-private-genad_improved_yjz/sgm/modules/attention3d.py utils
def expand_first(feat, scale=1.):
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)

def concat_first(feat, dim=2, scale=1.):
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, dim=-2, eps: float = 1e-5):
    # feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    # feat_mean = feat.mean(dim=-2, keepdims=True)

    feat_std = (feat.var(dim=dim, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=dim, keepdims=True)

    return feat_mean, feat_std

def adain(feat):
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat

def apply_adain(x):
    # mini_test
    # x: b t hw c
    # Only hurt the last frame, verified
    # x[:, -1] = 0
    # return x

    # x: b t hw c
    n_cond = 2 # !! Hard coded
    x_cond, x_first, x_rest = x[:, :n_cond], x[:, n_cond:n_cond+1], x[:, n_cond+1:]
    # x_rest = torch.zeros_like(x_rest)
    # out = torch.cat([x_cond, x_first, x_rest], dim=1)
    # return out

    x_first_original = x_first.clone()
    x_rest_original = x_rest.clone()
    # x_cond: b 2 hw c
    # x_first: b 1 hw c
    # x_rest:  b T-3 hw c
    tgt_mean, tgt_std = calc_mean_std(x_rest, dim=-2)

    x_pred = torch.cat([x_first, x_rest], dim=1) # b T-2 hw c
    x_shift_left = x_pred[:, :-1] # b T-3 hw c
    x_shift_right = x_pred[:, 1:] # b T-3 hw c
    x_concat = torch.cat([x_shift_left, x_shift_right], dim=-2) # b T-3 2hw c
    ref_mean, ref_std = calc_mean_std(x_concat, dim=-2)

    # ! Use the first prediction frame
    # x_first = x_first.unsqueeze(1)  # b 1 1 hw c
    # x_rest  = x_rest.unsqueeze(1)   # b 1 T-3 hw c
    # x_first_rep = x_first.repeat(1, 1, x_rest.shape[2], 1, 1) # b 1 T-3 hw c

    # x_rest_cat = torch.cat([x_first_rep, x_rest], dim=1) # b 2 T-3 hw c
    # x_rest_cat = x_rest_cat.transpose(1, 2) # b T-3 2 hw c
    
    # x_rest_cat = rearrange(x_rest_cat, 'b t k s c -> b t (k s) c')  # b T-3 2hw c
    # ref_mean, ref_std = calc_mean_std(x_rest_cat, dim=-2)
    
    x_rest = (x_rest_original - tgt_mean) / tgt_std
    x_rest = x_rest * ref_std + ref_mean

    # out1 = torch.cat([x_cond, x_first_original, torch.zeros_like(x_rest)], dim=1)
    out2 = torch.cat([x_cond, x_first_original, x_rest], dim=1)
    # diff = (out1 - out2)

    # import pdb; pdb.set_trace()

    return out2
    # return out

### end GenAD-private-genad_improved_yjz/sgm/modules/attention3d.py utils

# modified following GenAD-private-genad_improved_yjz/sgm/modules/attention3d.py
class BasicSpatialTemporalTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, 
            dim, 
            n_heads, 
            d_head, 
            dropout=0., 
            context_dim=None, 
            gated_ff=True, 
            checkpoint=True,
            disable_self_attn=False, 
            attn_mode="softmax",
            num_frames=1,
            concat_cond=False,
            causal_temp=False,
            assist_attn=False,
            criss_cross=False,
            use_relative_position=True,
            zero_init_temporal=True):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, checkpoint=False)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout, checkpoint=False)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

        ### start temporal from GenAD
        self.num_frames = num_frames

        # TODO: redesign the network structure
        if num_frames > 1:  # build temporal attention layers
            # self.temporal_ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
            # self.temporal_ff_norm = nn.LayerNorm(dim)

            # 1st temporal self-attn
            self.temporal_attn1 = TemporalAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                                    temporal_length=num_frames,
                                                    use_relative_position=use_relative_position,
                                                    zero_init=zero_init_temporal)
            # 2nd temporal self-attn
            self.temporal_attn2 = TemporalAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                                    temporal_length=num_frames,
                                                    use_relative_position=use_relative_position,
                                                    zero_init=zero_init_temporal)
            # 3rd temporal self-attn
            self.temporal_attn3 = TemporalAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                                    temporal_length=num_frames,
                                                    use_relative_position=use_relative_position,
                                                    zero_init=zero_init_temporal)
            self.temporal_norm1 = nn.LayerNorm(dim)
            self.temporal_norm2 = nn.LayerNorm(dim)
            self.temporal_norm3 = nn.LayerNorm(dim)

            self.temporal_mask = None
            if causal_temp:
                self.temporal_mask = torch.tril(torch.ones((num_frames, num_frames)))

            if concat_cond:
                self.temporal_cond_embed1 = nn.Embedding(1, dim)  # TODO: use nn.Parameter instead
                self.temporal_cond_embed2 = nn.Embedding(1, dim)
                self.temporal_cond_embed3 = nn.Embedding(1, dim)
                self.temporal_cond_gate1 = nn.Parameter(torch.tensor([0.]))  # TODO: just zero init the embedding
                self.temporal_cond_gate2 = nn.Parameter(torch.tensor([0.]))
                self.temporal_cond_gate3 = nn.Parameter(torch.tensor([0.]))
            
            if assist_attn:
                self.temporal_assist_attn_horizontal1 = attn_cls(
                    query_dim=dim,
                    context_dim=None,
                    heads=n_heads // 2,
                    dim_head=d_head,
                    dropout=dropout,
                    backend=sdp_backend,
                    zero_init=zero_init_temporal
                )  # is self-attn as context is None
                self.temporal_assist_horizontal_norm1 = nn.LayerNorm(dim)
                self.temporal_assist_attn_horizontal2 = attn_cls(
                    query_dim=dim,
                    context_dim=None,
                    heads=n_heads // 2,
                    dim_head=d_head,
                    dropout=dropout,
                    backend=sdp_backend,
                    zero_init=zero_init_temporal
                )  # is self-attn as context is None
                self.temporal_assist_horizontal_norm2 = nn.LayerNorm(dim)
                self.temporal_assist_attn_horizontal3 = attn_cls(
                    query_dim=dim,
                    context_dim=None,
                    heads=n_heads // 2,
                    dim_head=d_head,
                    dropout=dropout,
                    backend=sdp_backend,
                    zero_init=zero_init_temporal
                )  # is self-attn as context is None
                self.temporal_assist_horizontal_norm3 = nn.LayerNorm(dim)
                if criss_cross:
                    self.temporal_assist_attn_vertical1 = attn_cls(
                        query_dim=dim,
                        context_dim=None,
                        heads=n_heads // 2,
                        dim_head=d_head,
                        dropout=dropout,
                        backend=sdp_backend,
                        zero_init=zero_init_temporal
                    )  # is self-attn as context is None
                    self.temporal_assist_vertical_norm1 = nn.LayerNorm(dim)
                    self.temporal_assist_attn_vertical2 = attn_cls(
                        query_dim=dim,
                        context_dim=None,
                        heads=n_heads // 2,
                        dim_head=d_head,
                        dropout=dropout,
                        backend=sdp_backend,
                        zero_init=zero_init_temporal
                    )  # is self-attn as context is None
                    self.temporal_assist_vertical_norm2 = nn.LayerNorm(dim)
                    self.temporal_assist_attn_vertical3 = attn_cls(
                        query_dim=dim,
                        context_dim=None,
                        heads=n_heads // 2,
                        dim_head=d_head,
                        dropout=dropout,
                        backend=sdp_backend,
                        zero_init=zero_init_temporal
                    )  # is self-attn as context is None
                    self.temporal_assist_vertical_norm3 = nn.LayerNorm(dim)

        USE_ADAIN = False
        self.use_adaIN = USE_ADAIN
        if self.use_adaIN:
            print("!!!Applying AdaLN!!!!")
            

    # def forward(self, x, context=None, size=None, cond_mask=None):
    #     forward_inputs = (x, context) if cond_mask is None else (x, context, size, cond_mask)
    #     return checkpoint(self._forward, forward_inputs, self.parameters(), self.checkpoint)

    def forward(self, x, context=None, size=None, cond_mask=None):
        cond_mask = cond_mask if cond_mask is not None else torch.ones(x.shape[0], requires_grad=False).to(x.device).float()
        cond_mask = cond_mask.detach()
        h, w = size if size is not None else (x.shape[-2], x.shape[-1])
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.num_frames > 1:
            if hasattr(self, "temporal_cond_embed1"):
                x += (cond_mask @ self.temporal_cond_embed1.weight * self.temporal_cond_gate1.tanh()).unsqueeze(1)

            # TODO: add AdaLN here
            b = x.shape[0] // self.num_frames
            # x: (b t) hw c
            # AdaLN
            if self.use_adaIN:
                x = rearrange(x, '(b t) ... -> b t ...', t=self.num_frames)
                # import pdb; pdb.set_trace()
                x = apply_adain(x)
                x = rearrange(x, 'b t ... -> (b t) ...', t=self.num_frames)

            # 1st temporal self-attn
            # b = x.shape[0] // self.num_frames
            x = rearrange(x, "(b t) hw c -> (b hw) t c", t=self.num_frames)
            x = self.temporal_attn1(self.temporal_norm1(x), mask=self.temporal_mask) + x
            if hasattr(self, "temporal_assist_attn_horizontal1"):
                x = rearrange(x, "(b h w) t c -> (b t h) w c", b=b, h=h, w=w)
                x = self.temporal_assist_attn_horizontal1(self.temporal_assist_horizontal_norm1(x)) + x
                if hasattr(self, "temporal_assist_attn_vertical1"):
                    x = rearrange(x, "(b h) w c -> (b w) h c", h=h, w=w)
                    x = self.temporal_assist_attn_vertical1(self.temporal_assist_vertical_norm1(x)) + x
                    x = rearrange(x, "(b w) h c -> b (h w) c", h=h, w=w)
                else:
                    x = rearrange(x, "(b h) w c -> b (h w) c", h=h, w=w)
            else:
                x = rearrange(x, "(b hw) t c -> (b t) hw c", b=b)

        # spatial self-attention
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x

        if self.num_frames > 1:
            if hasattr(self, "temporal_cond_embed2"):
                x += (cond_mask @ self.temporal_cond_embed2.weight * self.temporal_cond_gate2.tanh()).unsqueeze(1)

            # 2nd temporal self-attn
            b = x.shape[0] // self.num_frames
            x = rearrange(x, "(b t) hw c -> (b hw) t c", t=self.num_frames)
            x = self.temporal_attn2(self.temporal_norm2(x), mask=self.temporal_mask) + x
            if hasattr(self, "temporal_assist_attn_horizontal2"):
                x = rearrange(x, "(b h w) t c -> (b t h) w c", b=b, h=h, w=w)
                x = self.temporal_assist_attn_horizontal2(self.temporal_assist_horizontal_norm2(x)) + x
                if hasattr(self, "temporal_assist_attn_vertical2"):
                    x = rearrange(x, "(b h) w c -> (b w) h c", h=h, w=w)
                    x = self.temporal_assist_attn_vertical2(self.temporal_assist_vertical_norm2(x)) + x
                    x = rearrange(x, "(b w) h c -> b (h w) c", h=h, w=w)
                else:
                    x = rearrange(x, "(b h) w c -> b (h w) c", h=h, w=w)
            else:
                x = rearrange(x, "(b hw) t c -> (b t) hw c", b=b)

        # spatial self-attention
        x = self.attn2(self.norm2(x), context=context) + x

        if self.num_frames > 1:
            if hasattr(self, "temporal_cond_embed3"):
                x += (cond_mask @ self.temporal_cond_embed3.weight * self.temporal_cond_gate3.tanh()).unsqueeze(1)

            # 3rd temporal self-attn
            b = x.shape[0] // self.num_frames
            x = rearrange(x, "(b t) hw c -> (b hw) t c", t=self.num_frames)
            x = self.temporal_attn3(self.temporal_norm3(x), mask=self.temporal_mask) + x
            if hasattr(self, "temporal_assist_attn_horizontal3"):
                x = rearrange(x, "(b h w) t c -> (b t h) w c", b=b, h=h, w=w)
                x = self.temporal_assist_attn_horizontal3(self.temporal_assist_horizontal_norm3(x)) + x
                if hasattr(self, "temporal_assist_attn_vertical3"):
                    x = rearrange(x, "(b h) w c -> (b w) h c", h=h, w=w)
                    x = self.temporal_assist_attn_vertical3(self.temporal_assist_vertical_norm3(x)) + x
                    x = rearrange(x, "(b w) h c -> b (h w) c", h=h, w=w)
                else:
                    x = rearrange(x, "(b h) w c -> b (h w) c", h=h, w=w)
            else:
                x = rearrange(x, "(b hw) t c -> (b t) hw c", b=b)

        # spatial self-attention
        x = self.ff(self.norm3(x)) + x
        # x = x.rearrange("b (h w) c -> b c h w", h=h, w=w)
        return x

# modified following GenAD-private-genad_improved_yjz/sgm/modules/attention3d.py
class SpatialTemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(
            self, 
            in_channels, 
            n_heads, 
            d_head,    
            depth=1, 
            dropout=0., 
            context_dim=None,
            disable_self_attn=False, 
            use_linear=False,
            attn_type="softmax",
            use_checkpoint=True, 
            num_frames=1,
            concat_cond=False,
            causal_temp=False,
            assist_attn=False,
            criss_cross=False
            ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicSpatialTemporalTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    attn_mode=attn_type,
                    checkpoint=use_checkpoint,
                    num_frames=num_frames,
                    concat_cond=concat_cond,
                    causal_temp=causal_temp,
                    assist_attn=assist_attn,
                    criss_cross=criss_cross
                )
                for d in range(depth)
            ]
        )
        # self.transformer_blocks = nn.ModuleList(
        #     [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
        #                            disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
        #         for d in range(depth)]
        # )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, cond_mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # cond_mask = cond_mask if cond_mask is not None else torch.ones(x.shape[0], requires_grad=False).to(x.device).float()
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if i > 0 and len(context) == 1:
                i = 0  # use same context for each block
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
            x = block(x, context=context[i])
            # h, w = x.shape[-2:]
            # x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w).contiguous()
            # x = block(x, context=context[i], size=(h, w), cond_mask=cond_mask)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
