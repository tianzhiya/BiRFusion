
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .layers import *
import kornia.utils as KU
import kornia.filters as KF
from copy import deepcopy
import os
from .irnn import irnn
os.environ['CUDA_VISIBLE_DEVICES']='0'

class SE(nn.Module):
    def __init__(self, in_dim=2048, sr_ratio=1):
        super(SE, self).__init__()
        input_dim = in_dim
        self.chanel_in = input_dim

        self.query_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.query_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.sr_ratio = sr_ratio
        dim = in_dim

        if sr_ratio > 1:
            self.sr_k = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_v = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_k = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_v = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

            self.sr_kk = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_vv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_kk = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_vv = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

        self.gamma_rd = nn.Parameter(torch.zeros(1))
        self.gamma_dr = nn.Parameter(torch.zeros(1))
        self.gamma_x = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(dim * 2, dim // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(dim // 2, dim * 2, kernel_size=1)
        self.merge_conv1x1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1), self.relu)

    def forward(self, xr, xd):
        # xr, xd = x[0].unsqueeze(dim=0), x[1].unsqueeze(dim=0)
        m_batchsize, C, width, height = xr.size()

        out_rd = xr
        out_dr = xd
        rgb_gap = nn.AvgPool2d(out_rd.shape[2:])(out_rd).view(len(out_rd), C, 1, 1)
        hha_gap = nn.AvgPool2d(out_dr.shape[2:])(out_dr).view(len(out_dr), C, 1, 1)
        stack_gap = torch.cat([rgb_gap, hha_gap], dim=1)
        stack_gap = self.fc1(stack_gap)
        stack_gap = self.relu(stack_gap)
        stack_gap = self.fc2(stack_gap)
        rgb_ = stack_gap[:, 0:C, :, :] * out_rd
        hha_ = stack_gap[:, C:2 * C, :, :] * out_dr
        merge_feature = torch.cat([rgb_, hha_], dim=1)
        merge_feature = self.merge_conv1x1(merge_feature)

        rgb_out = (xr + merge_feature) / 2
        hha_out = (xd + merge_feature) / 2
        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)

        # out_x = torch.cat([rgb_out, hha_out], dim=0)

        return rgb_out, hha_out

class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=False)


###############################
class Qkv(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x).cuda()
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear').cuda()
        actv = self.mlp_shared(segmap).cuda()
        gamma = self.mlp_gamma(actv).cuda()
        beta = self.mlp_beta(actv).cuda()
        out = self.bn(normalized * (1 + gamma)) + beta
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)  # First fully connected layer
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)  # Second fully connected layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global average pooling
        batch_size, channels, _, _ = x.size()
        squeezed = F.adaptive_avg_pool2d(x, (1, 1))
        squeezed = squeezed.view(batch_size, channels)

        # Excitation: Two fully connected layers with ReLU and Sigmoid
        excitation = F.relu(self.fc1(squeezed))
        excitation = self.sigmoid(self.fc2(excitation))

        # Reshape the excitation to match the input shape
        excitation = excitation.view(batch_size, channels, 1, 1)

        # Scale the input feature map by the excitation
        return x * excitation

class DispEstimator(nn.Module):
    def __init__(self,channel,depth=4,norm=nn.BatchNorm2d,dilation=1):
        super(DispEstimator,self).__init__()
        estimator = nn.ModuleList([])
        self.corrks = 7
        self.preprocessor = Conv2d(channel,channel,3,act=None,norm=None,dilation=dilation,padding=dilation)
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,act=None))
        #self.localcorrpropcessor = nn.Sequential(Conv2d(self.corrks**2,32,3,padding=1,bias=True,norm=None),
        #                                         Conv2d(32,2,3,padding=1,bias=True,norm=None),)
        oc = channel
        ic = channel+self.corrks**2
        dilation = 1
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.layers = estimator
        self.scale = torch.FloatTensor([256,256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
        #self.corrpropcessor = Conv2d(9+channel,channel,3,padding=1,bias=True,norm=nn.InstanceNorm2d)
        #self.AP3=nn.AvgPool2d(3,stride=1,padding=1)


    def localcorr(self,feat1,feat2):
        feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
        b,c,h,w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1,(13,13),(3,3),border_type='constant')
        feat1_loc_blk = F.unfold(feat1_smooth,kernel_size=self.corrks,dilation=4,padding=2*(self.corrks-1),stride=1).reshape(b,c,-1,h,w)
        localcorr = (feat2.unsqueeze(2)-feat1_loc_blk).pow(2).mean(dim=1)

        feat = feat.cuda()

        # corr = torch.cat([feat, localcorr], dim=1)
        B, C2, H, W = localcorr.shape
        C1 = feat.shape[1]

        # 如果通道不一致，使用1x1卷积调整 feat 的通道为 C2
        if C1 != C2:
            conv1x1 = nn.Conv2d(C1, C2, kernel_size=1).to(feat.device)
            feat1 = conv1x1(feat)

        qkv = Qkv(C2, C2, nhidden=64).cuda()
        corr = qkv(localcorr, feat1).cuda()
        corr = torch.cat([feat, corr], dim=1)

        return corr

    def forward(self,feat1,feat2):
        b,c,h,w = feat1.shape
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat1 = feat[:b]
        feat2 = feat[b:]
        if self.scale[0,1,0,0] != w-1 or self.scale[0,0,0,0] != h-1:
            self.scale = torch.FloatTensor([w,h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
            self.scale = self.scale.to(feat1.device)
        corr = self.localcorr(feat1,feat2)
        for i,layer in enumerate(self.layers):
            corr = layer(corr)
        corr = KF.gaussian_blur2d(corr,(13,13),(3,3),border_type='replicate')
        disp = corr.clamp(min=-300,max=300)
        # print(disp.shape)
        # print(feat1.shape)
        return disp/self.scale

class DispRefiner(nn.Module):
    def __init__(self,channel,dilation=1,depth=4):
        super(DispRefiner,self).__init__()
        self.preprocessor = nn.Sequential(Conv2d(channel,channel,3,dilation=dilation,padding=dilation,norm=None,act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,norm=None,act=None))
        oc = channel
        ic = channel+2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)
    def forward(self,feat1,feat2,disp):
        
        b=feat1.shape[0]
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b],feat[b:]],dim=1))
        corr = torch.cat([feat,disp],dim=1)
        delta_disp = self.estimator(corr)
        disp = disp+delta_disp
        return disp 
        

class Feature_extractor_unshare(nn.Module):
    def __init__(self,depth,base_ic,base_oc,base_dilation,norm):
        super(Feature_extractor_unshare,self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i%2==1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(ResConv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            else:
                feature_extractor.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            if i%2==1 and i<depth-1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv  # 使用图卷积层
class GNNMatcher(nn.Module):
    def __init__(self, unshare_depth=4, matcher_depth=4, num_pyramids=2, hidden_dim=64):
        super(GNNMatcher, self).__init__()
        self.num_pyramids = num_pyramids
        self.hidden_dim = hidden_dim

        # 原特征提取器部分仍然保持为传统的卷积网络来进行初步的特征提取
        self.feature_extractor_unshare1 = Feature_extractor_unshare(depth=unshare_depth, base_ic=3, base_oc=8,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)
        self.feature_extractor_unshare2 = Feature_extractor_unshare(depth=unshare_depth, base_ic=3, base_oc=8,
                                                                    base_dilation=1, norm=nn.InstanceNorm2d)

        # 使用图卷积网络进行后续的特征处理
        self.gnn_layer1 = GCNConv(8, hidden_dim)  # GCN 图卷积层
        self.gnn_layer2 = GCNConv(hidden_dim, hidden_dim)
        self.gnn_layer3 = GCNConv(hidden_dim, 1)  # 输出配对结果

    def construct_graph(self, feat1, feat2):
        """
        根据特征图生成图结构，构建节点和边的关系
        feat1: 图像1的特征图
        feat2: 图像2的特征图
        """
        # 将图像特征展平为节点
        node1 = feat1.view(feat1.size(0), -1)  # (B, C, H, W) -> (B, C*H*W)
        node2 = feat2.view(feat2.size(0), -1)

        # 创建节点间的边，基于某些距离度量或相似度
        edge_index = self.create_edges(node1, node2)

        return node1, node2, edge_index

    def create_edges(self, node1, node2):
        # 这里是一个简单的全连接图，实际中你可以用像素间的距离或相似度来创建边
        # 返回图的边结构
        num_nodes = node1.size(1)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().to(node1.device)  # 完全连接图
        return edge_index

    def match(self, feat11, feat12, feat21, feat22):
        # 将特征图转化为图结构
        node1, node2, edge_index = self.construct_graph(feat11, feat12)

        # 使用图卷积网络进行特征匹配
        out1 = F.relu(self.gnn_layer1(node1, edge_index))
        out1 = F.relu(self.gnn_layer2(out1, edge_index))

        out2 = F.relu(self.gnn_layer1(node2, edge_index))
        out2 = F.relu(self.gnn_layer2(out2, edge_index))

        # 计算节点之间的匹配
        disp = self.gnn_layer3(out1, edge_index) - self.gnn_layer3(out2, edge_index)

        return disp

    def forward(self, src, tgt, type='ir2vis'):
        b, c, h, w = tgt.shape

        # 提取图像的特征
        feat01 = self.feature_extractor_unshare1(src)
        feat02 = self.feature_extractor_unshare2(tgt)

        # 进行特征匹配
        disp_12 = self.match(feat01, feat02, feat01, feat02)  # 示例中使用相同的特征进行匹配

        if type == 'ir2vis':
            disp_12 = F.interpolate(disp_12, [h, w], mode='bilinear')
        elif type == 'vis2ir':
            disp_12 = F.interpolate(disp_12, [h, w], mode='bilinear')

        return {'ir2vis': disp_12}



class EncodeDecode(nn.Module):
    def __init__(self,unshare_depth=4,matcher_depth=4,num_pyramids=2):
        super(EncodeDecode, self).__init__()
        self.num_pyramids=num_pyramids
        self.encodeFeatureIr = Feature_extractor_unshare(depth=unshare_depth, base_ic=3, base_oc=8, base_dilation=1, norm=nn.InstanceNorm2d)
        self.encodeFeatureVis = Feature_extractor_unshare(depth=unshare_depth, base_ic=3, base_oc=8, base_dilation=1, norm=nn.InstanceNorm2d)
        #self.feature_extractor_unshare2 = self.feature_extractor_unshare1
        base_ic = self.encodeFeatureIr.ic
        base_oc = self.encodeFeatureIr.oc
        base_dilation = self.encodeFeatureIr.dilation
        self.feature_extractor_share1 = nn.Sequential(Conv2d(base_oc,base_oc*2,kernel_size=3,stride=1,padding=1,dilation=1, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*2,base_oc*2,kernel_size=3,stride=2,padding=1,dilation=1, norm=nn.InstanceNorm2d))
        self.feature_extractor_share2 = nn.Sequential(Conv2d(base_oc*2,base_oc*4,kernel_size=3,stride=1,padding=2,dilation=2, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*4,base_oc*4,kernel_size=3,stride=2,padding=2,dilation=2, norm=nn.InstanceNorm2d))
        self.feature_extractor_share3 = nn.Sequential(Conv2d(base_oc*4,base_oc*8,kernel_size=3,stride=1,padding=4,dilation=4, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*8,base_oc*8,kernel_size=3,stride=2,padding=4,dilation=4, norm=nn.InstanceNorm2d))
        self.matcher1 = DispEstimator(32,matcher_depth,dilation=4)
        self.matcher2 = DispEstimator(base_oc*8,matcher_depth,dilation=2)
        self.refiner = DispRefiner(base_oc*2,1)
        self.grid_down = KU.create_meshgrid(64,64).cuda()
        self.grid_full = KU.create_meshgrid(128,128).cuda()
        self.scale = torch.FloatTensor([128,128]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
        self.sE1=SE(in_dim=32,sr_ratio=2)
        self.sE2=SE(in_dim=128,sr_ratio=2)

    def match(self, feat11, feat12, feat31, feat32):
        # compute scale (w,h)
        if self.scale[0, 1, 0, 0] * 2 != feat11.shape[2] - 1 or self.scale[0, 0, 0, 0] * 2 != feat11.shape[3] - 1:
            self.h, self.w = feat11.shape[2], feat11.shape[3]
            self.scale = torch.FloatTensor([self.w, self.h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) - 1
            self.scale = self.scale.to(feat11.device)

        # estimate initial disparity from low-res features
        disp2_raw = self.matcher2(feat31, feat32)

        # upsample disp2 to feat11 resolution
        disp2 = F.interpolate(disp2_raw, [feat11.shape[2], feat11.shape[3]], mode='bilinear')

        # create or update full-resolution grid
        if disp2.shape[2] != self.grid_full.shape[1] or disp2.shape[3] != self.grid_full.shape[2]:
            self.grid_full = KU.create_meshgrid(feat11.shape[2], feat11.shape[3]).cuda()

        # warp feat11 using upsampled disp2
        feat11_warped = F.grid_sample(feat11, self.grid_full + disp2.permute(0, 2, 3, 1))

        # estimate finer disparity using warped feat11 and feat12
        disp1_raw = self.matcher1(feat11_warped, feat12)

        # upsample disp1 if necessary (should be same size here)
        disp1 = F.interpolate(disp1_raw, [feat11.shape[2], feat11.shape[3]], mode='bilinear')

        # refine
        disp_scaleup = (disp1 + disp2) * self.scale
        disp = self.refiner(feat11_warped, feat12, disp_scaleup)
        disp = KF.gaussian_blur2d(disp, (17, 17), (5, 5), border_type='replicate') / self.scale


        if self.training:
            return disp, disp_scaleup / self.scale, disp2
        return disp, None, None

    def forward(self,src,tgt,type='ir2vis'):
        b,c,h,w = tgt.shape
        feat01 = self.encodeFeatureIr(src)
        feat02 = self.encodeFeatureVis(tgt)
        feat0 = torch.cat([feat01,feat02])
        feat1 = self.feature_extractor_share1(feat0)
        feat2 = self.feature_extractor_share2(feat1)
        feat3 = self.feature_extractor_share3(feat2)
        feat11,feat12 = feat1[0:b],feat1[b:]
        feat31,feat32 = feat3[0:b],feat3[b:]
        disp_12 = None
        disp_21 = None

        feat11, feat12=self.sE1(feat11, feat12)
        feat31, feat32=self.sE2(feat31, feat32)

        if type == 'bi':
            disp_12,disp_12_down4,disp_12_down8 = self.match(feat11,feat12,feat31,feat32)
            disp_21,disp_21_down4,disp_21_down8 = self.match(feat12,feat11,feat32,feat31)
            t = torch.cat([disp_12,disp_21,disp_12_down4,disp_21_down4,disp_12_down8,disp_21_down8])
            t = F.interpolate(t,[h,w],mode='bilinear')
            down2,down4,donw8 = torch.split(t,2*b,dim=0)
            disp_12_,disp_21_ = torch.split(down2,b,dim=0)
        elif type == 'ir2vis':
            disp_12,_,_= self.match(feat11,feat12,feat31,feat32)
            disp_12 = F.interpolate(disp_12,[h,w],mode='bilinear')
        elif type =='vis2ir':
            disp_21,_,_ = self.match(feat12,feat11,feat32,feat31)
            disp_21 = F.interpolate(disp_21,[h,w],mode='bilinear')
        if self.training:
            return {'ir2vis':disp_12_,'vis2ir':disp_21_}
        return {'ir2vis':disp_12,'vis2ir':disp_21}

class Spacial_IRNN(nn.Module):
    def __init__(self, in_channels, alpha=0.2):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.IRNN = irnn()

    def forward(self, input):
        output = self.IRNN.apply(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight,
                      self.left_weight.weight, self.up_weight.bias, self.right_weight.bias, self.down_weight.bias,
                      self.left_weight.bias)
        return output


def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / \
                float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass


if __name__ == '__main__':
    matcher = EncodeDecode().cuda()
    ir = torch.rand(2,3,512,512).cuda()
    vis = torch.rand(2,3,512,512).cuda()
    disp=matcher(ir,vis,'bi')
