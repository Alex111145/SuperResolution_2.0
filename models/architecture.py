import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# -------------------------------------------------------------------------
# 1. MODULI BASE (Swin Transformer)
# -------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=None)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm: x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        return x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'Scala {scale} non supportata. Usa potenze di 2 o 3.')
        super(Upsample, self).__init__(*m)

# -------------------------------------------------------------------------
# 2. MODULI REAL-ESRGAN (RRDB - Residual in Residual Dense Block)
# -------------------------------------------------------------------------

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, nf: number of features
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Inizializzazione per stabilità
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # Residual scaling

class RRDB(nn.Module):
    """Residual in Residual Dense Block (Cuore di ESRGAN)."""
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x  # Residual scaling

# -------------------------------------------------------------------------
# 3. MODELLO IBRIDO (SwinIR + Real-ESRGAN)
# -------------------------------------------------------------------------

class HybridSwinRRDB(nn.Module):
    """
    MODELLO IBRIDO: 
    Usa Swin Transformer per la struttura globale e RRDB (ESRGAN) per le texture.
    """
    def __init__(self, img_size=64, in_chans=1, embed_dim=96, depths=[6, 6, 6], 
                 num_heads=[6, 6, 6], window_size=7, upscale=2, 
                 num_rrdb=3,  # Numero di blocchi RRDB da inserire dopo Swin
                 **kwargs):
        super(HybridSwinRRDB, self).__init__()
        
        self.upscale = upscale
        self.window_size = window_size
        self.embed_dim = embed_dim

        # 1. Feature Extraction (Convolutional)
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep Feature Extraction (Swin Transformer Body)
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)
        
        self.swin_layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim, 
                    input_resolution=(img_size, img_size),
                    num_heads=num_heads[i], 
                    window_size=window_size,
                    shift_size=0 if (j % 2 == 0) else window_size // 2
                ) for j in range(depths[i])
            ])
            self.swin_layers.append(layer)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_swin = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. Texture Refinement (Real-ESRGAN / RRDB Body)
        # Inseriamo i blocchi RRDB qui per raffinare le features estratte dal Transformer
        # prima di mandarle all'upsampler.
        self.rrdb_blocks = nn.Sequential(*[
            RRDB(nf=embed_dim, gc=32) for _ in range(num_rrdb)
        ])
        
        # Convoluzione post-RRDB
        self.conv_after_rrdb = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 4. Upsampling & Reconstruction
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample = Upsample(upscale, 64)
        self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)

    def forward(self, x):
        # Gestione padding dinamico (evita crash se dimensioni non divisibili per window_size)
        H, W = x.shape[2], x.shape[3]
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # --- Stage 1: Conv ---
        x_first = self.conv_first(x_padded)
        
        # --- Stage 2: Swin Transformer (Global Features) ---
        res = self.patch_embed(x_first)
        for layer in self.swin_layers:
            for block in layer:
                res = block(res)
        res = self.norm(res)
        res = self.patch_unembed(res, (x_padded.shape[2], x_padded.shape[3]))
        
        # Skip connection globale Swin
        x_swin = self.conv_after_swin(res) + x_first
        
        # --- Stage 3: RRDB (Local Texture Refinement) ---
        # Il Transformer ha fatto il lavoro grosso sulla struttura.
        # Ora l'RRDB lavora sui dettagli ad alta frequenza.
        x_rrdb = self.rrdb_blocks(x_swin)
        x_refined = self.conv_after_rrdb(x_rrdb) + x_swin  # Residual connection anche qui
        
        # --- Stage 4: Upscale ---
        out = self.conv_before_upsample(x_refined)
        out = self.upsample(out)
        out = self.conv_last(out)
        
        # Crop finale per tornare alle dimensioni target esatte
        return out[:, :, :H*self.upscale, :W*self.upscale]

# Alias per compatibilità se richiami ancora SwinIR ma vuoi usare l'ibrido
# Puoi cambiare questa riga in `SwinIR = HybridSwinRRDB` se vuoi sostituirlo ovunque
class SwinIR(HybridSwinRRDB):
    pass
