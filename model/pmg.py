import torch
import torch.nn as nn
import torch.nn.functional as F
from model.text_guidance import LightweightImageEncoder, MultiHeadCrossAttention, AdaLN3d
class discriminator(nn.Module):

    def __init__(self, inchannel, outchannel, num_classes, patch_size):
        super(discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(inchannel, 64, kernel_size=3, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(dim, num_classes)
        # self.p_mu = nn.Linear(dim, outchannel, nn.LeakyReLU())
        self.pro_head = nn.Linear(dim, outchannel, nn.ReLU())
        self.clip_pro_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 512),
        )

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel, self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.mp(self.relu1(self.conv1(x)))
            out2 = self.mp(self.relu2(self.conv2(out1)))
            out2 = out2.view(in_size, -1)
            w, h = out2.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x, mode='test'): 

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            proj = self.pro_head(out4)
            proj_norm = F.normalize(proj)
            clip_proj = F.normalize(self.clip_pro_head(out4))
            clss = self.cls_head_src(out4)

            return clss, proj_norm, clip_proj


class Spa_Spe_Randomization(nn.Module):
    def __init__(self, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(device)  

    def forward(self, x, ):
        N, C, L, H, W = x.size()
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()
        #doubt
            idx_swap = torch.randperm(N)  
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]  
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, L, H, W)

        return x, idx_swap


class Generator_3DCNN_SupCompress_pca(nn.Module):

    def __init__(self, imdim=3, imsize=[13, 13], device=0, dim1=128, dim2=8, text_dim=0, no_cross_attention=False, no_adaln=False, no_semantic_guidance=False):
        super().__init__()

        self.patch_size = imsize[0]
        self.n_channel = dim2
        self.n_pca = dim1
        self.no_cross_attention = no_cross_attention
        self.no_adaln = no_adaln
        self.no_semantic_guidance = no_semantic_guidance
        self.text_guided = (text_dim > 0) and (not no_semantic_guidance)

        # 2D_CONV
        self.conv_pca = nn.Conv2d(imdim, self.n_pca, 1, 1) 
        self.inchannel = self.n_pca

        # 3D_CONV
        self.conv1 = nn.Conv3d(in_channels=1,
                               out_channels=self.n_channel,
                               kernel_size=(3, 3, 3))

        # Style mixing/AdaLN conditioning
        if self.text_guided:
            self.image_encoder = LightweightImageEncoder(in_channels=imdim)
            self.cross_attention = MultiHeadCrossAttention(img_dim=imdim, text_dim=text_dim)
            self.alpha = nn.Parameter(torch.tensor(0.1))
            self.ada_ln = AdaLN3d(n_channel=self.n_channel, text_dim=imdim) # Conditioned by Semantic Representation (dimension=imdim)
            self.Spa_Spe_Random = None
        else:
            self.image_encoder = None
            self.cross_attention = None
            self.alpha = None
            self.ada_ln = None
            self.Spa_Spe_Random = Spa_Spe_Randomization(device=device)

        # 3D transpose & output Conv2D
        self.conv6 = nn.ConvTranspose3d(in_channels=self.n_channel, out_channels=1, kernel_size=(3, 3, 3))
        self.conv_inverse_pca = nn.Conv2d(self.n_pca, imdim, 1, 1)

    def forward(self, x, text_cond=None, all_class_features=None):
        if self.text_guided:
            # 1. Lightweight Image Encoder
            img_feats = self.image_encoder(x)
            
            # 2. Cross Attention (Q=Image Feature, K=Prompt Embeddings, V=Prompt Embeddings)
            if self.no_cross_attention:
                sem_rep = img_feats
            else:
                cross_attn_cond = all_class_features if all_class_features is not None else text_cond
                sem_rep = self.cross_attention(img_feats, cross_attn_cond)
            
            # 3. Fusion (Image Feature + Semantic Representation)
            x_fused = img_feats + self.alpha * sem_rep
            
            # 4. Backbone forward
            x_conv = self.conv_pca(x_fused)
            x_3d = x_conv.reshape(-1, self.patch_size, self.patch_size, self.inchannel, 1)
            x_3d = x_3d.permute(0, 4, 3, 1, 2)
            x_3d = F.relu(self.conv1(x_3d))
            
            # AdaLN conditioned by Semantic Representation (pooled to 1D vector per sample)
            if self.no_adaln:
                x_3d = self.ada_ln(x_3d, None)
            else:
                sem_cond = F.adaptive_avg_pool2d(sem_rep, 1).flatten(1)
                x_3d = self.ada_ln(x_3d, sem_cond)
            
            x_3d = torch.sigmoid(self.conv6(x_3d))
            x_out = x_3d.permute(0, 2, 3, 4, 1)
            x_out = x_out.reshape(-1, self.inchannel, self.patch_size, self.patch_size)
            x_out = self.conv_inverse_pca(x_out)
            return x_out
        else:
            x_conv = self.conv_pca(x)
            x_3d = x_conv.reshape(-1, self.patch_size, self.patch_size, self.inchannel, 1)
            x_3d = x_3d.permute(0, 4, 3, 1, 2)
            x_3d = F.relu(self.conv1(x_3d))
            if self.Spa_Spe_Random is not None:
                x_3d, _ = self.Spa_Spe_Random(x_3d)
            x_3d = torch.sigmoid(self.conv6(x_3d))
            x_out = x_3d.permute(0, 2, 3, 4, 1)
            x_out = x_out.reshape(-1, self.inchannel, self.patch_size, self.patch_size)
            x_out = self.conv_inverse_pca(x_out)
            return x_out
    
def downsample(img, m):
    b, total_channels, height, width = img.shape
    
    
    group_channels = total_channels // m
    remainder = total_channels % m
    
  
    reduced_img = torch.zeros(b, m, height, width, device=img.device, dtype=img.dtype)
    end_channel = -1

    for i in range(m):
        start_channel = end_channel+1
        end_channel = start_channel + group_channels-1 + (1 if remainder > 0 else 0)
        remainder -= 1
        reduced_img[:, i, :, :] = img[:, start_channel:end_channel, :, :].mean(dim=1)
    return reduced_img
#doubt 
class Generator(nn.Module):
    def __init__(self, imdim=48, patch_size=13, layers = [], dim1 = 128, dim2 = 8, device=0, text_dim=0, no_cross_attention=False, no_adaln=False, no_semantic_guidance=False):
        super().__init__()
        self.patch_size = patch_size
        self.n_channel = imdim
        self.layers_num = len(layers)
        self.dims = layers
        self.conv_pcas = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.sub_g = nn.ModuleList()
        for i in range(self.layers_num-1):
            self.conv_pcas.append(nn.Conv2d(in_channels=self.n_channel, out_channels=self.dims[i], kernel_size=1))
        for i in range(self.layers_num-1):
            self.upsamples.append(nn.Conv2d(in_channels=self.dims[i], out_channels=self.dims[i+1], kernel_size=1) )
        for i in range(self.layers_num):
            self.sub_g.append(Generator_3DCNN_SupCompress_pca(
                imdim=self.dims[i], imsize=[self.patch_size, self.patch_size], device=device, dim1=dim1, dim2=dim2, text_dim=text_dim,
                no_cross_attention=no_cross_attention, no_adaln=no_adaln, no_semantic_guidance=no_semantic_guidance
            ))
        
        self.text_guided = (text_dim > 0) and (not no_semantic_guidance)

    def forward(self, x, current_step = 9999, text_features=None, all_class_features=None):
        if current_step <= self.layers_num:
            if current_step > 1:
                self.sub_g[current_step-2].requires_grad_(False)
                self.upsamples[current_step-2].requires_grad_(False)
                self.conv_pcas[current_step-2].requires_grad_(False)
            if current_step < self.layers_num:
                x_down = self.conv_pcas[current_step-1](x)
            else:
                x_down = x
                
            if len(self.dims) > 1:
                f_old = downsample(x, self.dims[0])
            else:
                f_old = x
                
            x_g = self.sub_g[0](f_old, text_cond=text_features, all_class_features=all_class_features)
            
            for i in range(1, current_step):
                f_old = self.upsamples[i-1](x_g)
                x_g = self.sub_g[i](f_old, text_cond=text_features, all_class_features=all_class_features)
            return x_g, x_down
        else:
            if current_step < self.layers_num*2:
                x_down = self.conv_pcas[current_step-1-self.layers_num](x)
            else:
                x_down = x
            return x_down
            
class Dis(nn.Module):
    def __init__(self, imdim=48, patch_size=13, layers = [], proj=128, num_classes=7):
        super().__init__()
        self.patch_size = patch_size
        self.n_channel = imdim
        self.layers_num = len(layers)
        self.dims = layers
        # self.dims = [int((imdim+layers_num)/layers_num)*(i+1) for i in range(layers_num-1)]+[imdim]
        self.sub_d = nn.ModuleList()
        for i in range(self.layers_num):
            self.sub_d.append(discriminator(inchannel=self.dims[i], outchannel=proj, num_classes=num_classes, patch_size=self.patch_size))
        
    def forward(self, x, current_step = 9999, mode='train'):
        
        if current_step <= self.layers_num:
            if current_step > 1:
                self.sub_d[current_step-2].requires_grad_(False)
            return self.sub_d[current_step-1](x, mode=mode)
        else:
            clss = self.sub_d[-1](x, mode='test')
            return clss
        
        