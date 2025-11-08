from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.checkpoint

class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop = 0., proj_drop = 0):
        super().__init__()
        # dim has been modified to the dim of each head (embedding)
        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.out_channel = dim * num_heads
        self.scale = dim **-0.5

        self.qkv = nn.Linear(self.out_channel, self.out_channel * 3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.out_channel, self.out_channel)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # batch, number of patches, Channel (actual patch embeding * heads * 3)
        B, N, C = x.shape
        assert C == self.out_channel, 'total dim of out_channel dismatch'
        # reshape into B N 3(QKV) heads(parallel attentions, H) each patch(P) 
        # note: H * P = C
        # after permute: 3 B H N P
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
        # B H N P
        q, k, v = qkv.unbind(0)

        # B H N N 
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        # B H N P then transpose B N H P then reshape: B N C
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values = 1e-5, inplace = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))


    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias =(bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio = 4., qkv_bias = False,
            drop = 0., attn_drop = 0, init_values = 1e-5, drop_path = 0.,
            act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()
        self.out_channel = dim * num_heads
        self.norm1 = norm_layer(self.out_channel)
        self.attn = Attention(dim, num_heads, qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop)
        self.ls1 = LayerScale(self.out_channel, init_values = init_values)
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(self.out_channel)
        self.mlp = Mlp(in_features=self.out_channel, hidden_features=int(self.out_channel * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(self.out_channel, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

# set stacks = 2 to embed expert and novice individually
class ModalityEmbed(nn.Module):
    def __init__(self, patch_frame = 1, modality_size=2, num_heads = 8, embed_dim=16, layer_norm = False, modality_stacks = 1):
        super().__init__()
        assert (modality_stacks == 1 or modality_stacks == 2)
        out_channel = num_heads * embed_dim
        self.layer_norm = layer_norm
        self.modality_stacks = modality_stacks
        self.norm = nn.LayerNorm(out_channel)
        self.proj = nn.Conv1d(modality_size, out_channel, kernel_size=patch_frame, stride=patch_frame)
        # for separate embeddings
        # use same embedding for both or separately? 
        # same one for now.
        # input is (B, in channel, L) output is (B, out channel, L out)
        self.proj1 = nn.Conv1d(modality_size // modality_stacks, out_channel, kernel_size=patch_frame, stride=patch_frame)
        # self.proj2 = nn.Conv1d(modality_size // modality_stacks, out_channel, kernel_size=patch_frame, stride=patch_frame)
        self.patch_frame = patch_frame
        self.modality_size = modality_size

    def forward(self, x):
        # Batch, number of frames, length of modality
        B, F, M = x.shape
        assert (self.modality_size % self.modality_stacks) == 0
        modality_unit_size = self.modality_size // self.modality_stacks
        assert (F % self.patch_frame) == 0
        assert (M % modality_unit_size) == 0
        
        # if stacks = 2, it will be exp1 nov1 exp2 nov2 ...
        if self.modality_stacks == 2:
            x1 = x[:,:, 0 : modality_unit_size]
            x2 = x[:,:, modality_unit_size : modality_unit_size * 2]
            x1 = x1.transpose(-2, -1)
            x2 = x2.transpose(-2, -1)
            x1 = self.proj1(x1)
            x2 = self.proj1(x2)
            # B, C, L -> B, L, C
            x1 = x1.transpose(-2,-1)
            x2 = x2.transpose(-2,-1)
            # both are B, L, C, since each person is embedded separately, total L is doubled.
            x = torch.cat((x1, x2), dim=1)

        else:
            # B, M, F
            x = x.transpose(-2, -1)
            # conv1d over frames, B, out_chennal (actual embeddings * heads), output lengths (number of embedings)
            x = self.proj(x)
            # B, C, L -> B, L, C
            x = x.transpose(1, 2)

        # layer norm
        if self.layer_norm:
            x = self.norm(x)
        return x


class SSTransformer(nn.Module):
    def __init__(
        self, window_frame, patch_frame, modality_size, num_heads = 8, embed_dim=16, layer_norm = False, depth = 6, drop_out = 0, qkv_bias = True, init_values = 1e-5, modality_stacks = 1):
        super().__init__()

        self.attn_out = 7
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        prefix_token = True

        self.patch_embed = ModalityEmbed(patch_frame = patch_frame, modality_size = modality_size, num_heads = num_heads, embed_dim = embed_dim, layer_norm = layer_norm, modality_stacks = modality_stacks)
        # heads * embed_dim = total dim for multi-head attation i.e. C in embed output (B, L, C)
        self.out_channel = num_heads * embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.out_channel))
        # actual number of embeddings + class 1 token, in this case it will be used for regression output.
        num_embed = modality_stacks * window_frame // patch_frame + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_embed, self.out_channel) * .02)
        self.pos_drop = nn.Dropout(p = drop_out)
        # self.drop_out = drop_out
        self.drop_out = drop_out
        dpr = [x.item() for x in torch.linspace(0, drop_out, depth)]  # stochastic depth decay rule

        self.blocks =  nn.Sequential(*[
            block(
                dim=embed_dim, num_heads = num_heads, mlp_ratio=4, qkv_bias=qkv_bias, init_values=init_values,
                drop= self.drop_out, attn_drop= drop_out, drop_path=dpr[i], norm_layer=norm_layer, act_layer = act_layer)
            for i in range(depth)])

        self.norm = norm_layer(self.out_channel)
        # no norm for class token

        # attn Head, output from each attn stream
        self.head = nn.Linear(self.out_channel, self.attn_out)

    def init_weight(self):
        return None

    # pos embeding AND prepend class token
    def _pos_embed(self, x):
        # -1 means not changing size
        # x is from embed (B, L, C)
        # cls token is (1, 1, C)
        
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_feature(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = x[:,0]
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        x = self.forward_head(x)
        return x

class MultiStreamSST(nn.Module):
    def __init__(self, window_frame, patch_frame, modality_size, num_heads = 8, embed_dim = [16, 16, 16, 16], layer_norm = False, depth = 6, drop_out = 0, qkv_bias = True, init_values = 1e-5, modality_stacks = 1) -> None:
        super().__init__()
        
        self.SST0 = SSTransformer(window_frame, patch_frame, modality_size[0], num_heads, embed_dim[0], layer_norm, depth, drop_out, qkv_bias, init_values, modality_stacks)
        self.SST1 = SSTransformer(window_frame, patch_frame, modality_size[1], num_heads, embed_dim[1], layer_norm, depth, drop_out, qkv_bias, init_values, modality_stacks)
        self.SST2 = SSTransformer(window_frame, patch_frame, modality_size[2], num_heads, embed_dim[2], layer_norm, depth, drop_out, qkv_bias, init_values, modality_stacks)
        self.SST3 = SSTransformer(window_frame, patch_frame, modality_size[3], num_heads, embed_dim[3], layer_norm, depth, drop_out, qkv_bias, init_values, modality_stacks)
        

        self.modality_divider = []
        tmp = 0
        for m in modality_size:
            tmp += m
            self.modality_divider.append(tmp)
        attn_out = 7
        num_attn = 4
        mlp_ratio = 4
        self.fc_norm = nn.LayerNorm(attn_out * num_attn)
        self.mlp = Mlp(in_features=attn_out * num_attn, hidden_features=int(attn_out * num_attn * mlp_ratio), out_features = 1)
        self.act = nn.Sigmoid()
    def regression_head(self, x):
        x = self.fc_norm(x)
        x = self.mlp(x)
        return self.act(x)
        # return torch.sigmoid(x)

    def forward(self, x):
        x0 = self.SST0(x[:,:,0:self.modality_divider[0]])
        x1 = self.SST1(x[:,:,self.modality_divider[0]:self.modality_divider[1]])
        x2 = self.SST2(x[:,:,self.modality_divider[1]:self.modality_divider[2]])
        x3 = self.SST3(x[:,:,self.modality_divider[2]:self.modality_divider[3]])
        x = torch.cat((x0, x1, x2, x3), dim=1)

        x = self.regression_head(x)
        return x

class alt_SSTransformer(nn.Module):
    def __init__(
        self, window_frame, patch_frame, num_heads = 8, embed_dim=16, depth = 6, drop_out = 0, qkv_bias = True, init_values = 1e-5, modality_stacks = 1):
        super().__init__()

        self.attn_out = 7
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU
        self.act = nn.GELU()
        embed_dim = embed_dim // 16
        # heads * embed_dim = total dim for multi-head attation i.e. C in embed output (B, L, C)
        self.out_channel = embed_dim * num_heads
        self.patch_embed = nn.Linear(self.out_channel * 2, self.out_channel)
        self.embed_mlp =  Mlp(in_features=self.out_channel * 2, hidden_features=int(self.out_channel * 2 * 2), out_features = self.out_channel)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.out_channel))
        # actual number of embeddings + class 1 token, in this case it will be used for regression output.
        num_embed = modality_stacks * window_frame // patch_frame + 1
        self.pos_embed = nn.Parameter(torch.randn(1, num_embed, self.out_channel) * .02)
        self.pos_drop = nn.Dropout(p = drop_out)
        # self.drop_out = drop_out
        self.drop_out = drop_out
        dpr = [x.item() for x in torch.linspace(0, drop_out, depth)]  # stochastic depth decay rule

        self.blocks =  nn.Sequential(*[
            block(
                dim=embed_dim, num_heads = num_heads, mlp_ratio=4, qkv_bias=qkv_bias, init_values=init_values,
                drop= self.drop_out, attn_drop= drop_out, drop_path=dpr[i], norm_layer=norm_layer, act_layer = act_layer)
            for i in range(depth)])

        self.norm = norm_layer(self.out_channel)
        # no norm for class token

        # attn Head, output from each attn stream
        self.head = nn.Linear(self.out_channel, self.attn_out)

    def init_weight(self):
        return None

    # pos embeding AND prepend class token
    def _pos_embed(self, x):
        # -1 means not changing size
        # x is from embed (B, L, C)
        # cls token is (1, 1, C)
        
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim = 1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_feature(self, x):
        # x = self.act(self.patch_embed(x))
        x = self.embed_mlp(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        x = x[:,0]
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        x = self.forward_head(x)
        return x

class alt_MultiStreamSST(nn.Module):
    def __init__(self, window_frame, patch_frame,num_heads = 8, embed_dim = [16, 16, 16, 16], layer_norm = False, depth = 6, drop_out = 0, qkv_bias = True, init_values = 1e-5, modality_stacks = 1) -> None:
        super().__init__()
        
        self.SST0 = alt_SSTransformer(window_frame, patch_frame, num_heads, embed_dim[0], depth, drop_out, qkv_bias, init_values, modality_stacks)
        self.SST1 = alt_SSTransformer(window_frame, patch_frame, num_heads, embed_dim[1], depth, drop_out, qkv_bias, init_values, modality_stacks)
        self.SST2 = alt_SSTransformer(window_frame, patch_frame, num_heads, embed_dim[2], depth, drop_out, qkv_bias, init_values, modality_stacks)
        self.SST3 = alt_SSTransformer(window_frame, patch_frame, num_heads, embed_dim[3], depth, drop_out, qkv_bias, init_values, modality_stacks)
        

        self.modality_divider = []
        tmp = 0
        for m in embed_dim:
            tmp += m
            self.modality_divider.append(tmp)
        attn_out = 7
        num_attn = 4
        mlp_ratio = 4
        self.fc_norm = nn.LayerNorm(attn_out * num_attn)
        self.mlp = Mlp(in_features=attn_out * num_attn, hidden_features=int(attn_out * num_attn * mlp_ratio), out_features = 1)
        self.act = nn.Sigmoid()
    def regression_head(self, x):
        x = self.fc_norm(x)
        x = self.mlp(x)
        return self.act(x)
        # return torch.sigmoid(x)

    def forward(self, x):
        x0 = self.SST0(x[:,:,0:self.modality_divider[0]])
        x1 = self.SST1(x[:,:,self.modality_divider[0]:self.modality_divider[1]])
        x2 = self.SST2(x[:,:,self.modality_divider[1]:self.modality_divider[2]])
        x3 = self.SST3(x[:,:,self.modality_divider[2]:self.modality_divider[3]])
        x = torch.cat((x0, x1, x2, x3), dim=1)

        x = self.regression_head(x)
        return x