import torch
from torch import nn
import torch.nn.functional as F
from loss import batch_episym
from config import get_config, print_usage
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU

class SSEU_Module(Module):
    """Split-Attention Conv2d   Groups=2   in_channels=128，   channels=128
     in_channels, channels  输入和输出通道 = group_width, group_width
     groups=cardinality =2

     self.conv2 = SSEU_Module(
                group_width, group_width, kernel_size=1,
                stride=stride,
                groups=cardinality, bias=True,
                radix=radix, rectify=rectified_conv,

                )
    """
    def __init__(self, in_channels, channels,  groups=2, bias=True, radix=2, reduction_factor=4, **kwargs):
        super(SSEU_Module, self).__init__()

        inter_channels = max(in_channels*radix//reduction_factor, 32)  # begin:128,here is 32
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.conv = Conv2d(in_channels, channels*radix, kernel_size=1, stride=1,  # 1*1？
                               groups=groups*radix, bias=bias, **kwargs)
        self.in0 = nn.InstanceNorm2d(channels*radix, eps=1e-5)
        self.bn0 = nn.BatchNorm2d(channels*radix)

        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        # self.in1 = nn.InstanceNorm2d(inter_channels, eps=1e-5) # we shold
        self.bn1 = nn.BatchNorm2d(inter_channels)

        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)

        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)   # firstly 1*1  group conv
        x = self.in0(x)    # add IN
        x = self.bn0(x)
        # if self.dropblock_prob > 0.0:  # take out
        #     x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1) # adaptive_avg_pool
        # print('gap1:',gap.shape,gap)
        gap = self.fc1(gap)  # 64

        gap = self.bn1(gap)
        # print('gap4:', gap.shape, gap)
        gap = self.relu(gap)  # activation function
        # print('gap5:', gap.shape, gap)
        atten = self.fc2(gap)   # finally we conv again
        # print('atten1:',atten.shape,atten)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1) # activation function
        # print('atten2:', atten.shape, atten)
        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel//self.radix), dim=1)
                # print('atten3:', atten.shape, atten)
            else:
                attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
            # print('out:', out.shape, out)
        else:
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class PESA_Block(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1,
                 radix=2, cardinality=2, bottleneck_width=64,
                 rectified_conv=False, rectify_avg=False):
        super(PESA_Block, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality # * cardinality
        print('radix:',radix,'groups:', cardinality)
        self.shot_cut = None
        if planes*2 != inplanes:
            self.shot_cut = nn.Conv2d(inplanes, planes*2, kernel_size=1)
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1,groups=cardinality*radix, bias=True)
        self.in1 = nn.InstanceNorm2d(group_width, eps=1e-5)   # eps=1e-3
        self.bn1 = nn.BatchNorm2d(group_width)
        self.radix = radix
        if radix >= 1:
            self.conv2 = SSEU_Module(
                group_width, group_width,
                groups=cardinality, bias=True,
                radix=radix
                )
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                groups=cardinality, bias=False)
            self.in2 = nn.InstanceNorm2d(group_width, eps=1e-5)
            self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, planes*2, kernel_size=1, bias=True)
        self.in3 = nn.InstanceNorm2d(planes*2, eps=1e-5)
        self.bn3 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.in1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.radix == 0:
            out = self.in2(out)
            out = self.bn2(out)
            out = self.relu(out)
        out = self.conv3(out)
        out = self.in3(out)
        out = self.bn3(out)

        if self.shot_cut:
            residual = self.shot_cut(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out


class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        # Spatial Correlation Layer
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out


class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x)# b*k*n*1
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_up, x_down):
        embed = self.conv(x_up)# b*k*n*1
        S = torch.softmax(embed, dim=1).squeeze(3)# b*k*n
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class PESA_subNet(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters, split, groups, model_name):
        nn.Module.__init__(self)
        channels = net_channels

        self.layer_num = depth
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num)+', split:'+str(split)+', groups:'+str(groups)+', '+model_name)
        # print('split_org_G1R2')
        self.conv1 = nn.Conv2d(input_channel, channels, kernel_size=1)
        self.in1 = nn.InstanceNorm2d(channels, eps=1e-5)
        self.bn1 = nn.BatchNorm2d(channels)
        self.Re = nn.ReLU(inplace=True)

        l2_nums = clusters 

        self.l1_1 = []
        for _ in range(self.layer_num//2):
            self.l1_1.append(PESA_Block(channels,channels//2,1,split, groups))# 1是setp

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(PESA_Block(2*channels, channels//2,1,split, groups))
        for _ in range(self.layer_num//2-1):
            self.l1_2.append(PESA_Block(channels,channels//2,1,split, groups))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)


    def forward(self, data, xs):
        #data: b*c*n*1
        batch_size, num_pts = data.shape[0], data.shape[2]
        x1_1 = self.conv1(data)
        x1_1 = self.l1_1(x1_1)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)
        x_up = self.up1(x1_1, x2)
        out = self.l1_2( torch.cat([x1_1,x_up], dim=1))
        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        e_hat = weighted_8points(xs, logits)
        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)
        return logits, e_hat, residual


class PESA_Net(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)
        self.weights_init = PESA_subNet(config.net_channels, 4+self.side_channel, depth_each_stage
                                     , config.clusters,config.split,config.group,config.model)
        self.weights_iter = [PESA_subNet(config.net_channels, 6+self.side_channel, depth_each_stage
                                      , config.clusters ,config.split,config.group,config.model) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data):

        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1

        input = data['xs'].transpose(1,3)   # input ������
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)
        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1),
                data['xs'])
            res_logits.append(logits), res_e_hat.append(e_hat)

        return res_logits, res_e_hat

        
def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat


#
# config, unparsed = get_config()
# print(OANet(config))