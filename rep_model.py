from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
# from itertools import product as product
import torchvision
from deform_conv2 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = 'cpu'


class VGGBase(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps
    Feel free to substitute with other pre-trained backbones
    """

    def __init__(self):
        super(VGGBase, self).__init__()

        # Standard convolutional layers in VGG16
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # stride = 1, by default
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # ceiling (not floor) here for even dims

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # retains size because stride is 1 (and padding)

        # Replacements for FC6 and FC7 in VGG16
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # atrous convolution

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        out = F.relu(self.conv1_1(image))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_3_feats = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19), pool5 does not reduce dimensions

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_feats = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        # Lower-level feature maps
        return conv4_3_feats, conv7_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("Load VGG-base model for RepPoint backbone. \n")


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # Auxiliary/additional convolutions on top of the VGG base, RepPoint heads are attached to each of the levels
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.conv8_2 = DeformConv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = DeformConv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        """
        Forward propagation.

        :param conv7_feats: lower-level conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :return: higher-level feature maps conv8_2, conv9_2, conv10_2, and conv11_2
        """
        out = F.relu(self.conv8_1(conv7_feats))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_2_feats = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_2_feats = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_2_feats = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_2_feats = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        # Higher-level feature maps
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 5685 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 5685 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes, n_points=9, center_init=True, transform_method='moment'):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes
        self.n_points = n_points
        self.gradient_multipler = 0.1
        self.center_init = center_init
        self.moment_multipler = 0.01
        self.transform_method = transform_method

        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv4_3': 3,
                   'conv7': 3,
                   'conv8_2': 2,
                   'conv9_2': 2,
                   'conv10_2': 2,
                   'conv11_2': 2}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions: initial reppoint locations
        self.loc_conv4_3_init = nn.Conv2d(512, n_boxes['conv4_3'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv7_init = nn.Conv2d(1024, n_boxes['conv7'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv8_2_init = nn.Conv2d(512, n_boxes['conv8_2'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv9_2_init = nn.Conv2d(256, n_boxes['conv9_2'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv10_2_init = nn.Conv2d(256, n_boxes['conv10_2'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv11_2_init = nn.Conv2d(256, n_boxes['conv11_2'] * self.n_points * 2, kernel_size=3, padding=1)

        # initial reppoint offsets for refinement
        self.loc_conv4_3_refine = DeformConv2d(512, n_boxes['conv4_3'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv7_refine = DeformConv2d(1024, n_boxes['conv7'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv8_2_refine = DeformConv2d(512, n_boxes['conv8_2'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv9_2_refine = DeformConv2d(256, n_boxes['conv9_2'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv10_2_refine = nn.Conv2d(256, n_boxes['conv10_2'] * self.n_points * 2, kernel_size=3, padding=1)
        self.loc_conv11_2_refine = nn.Conv2d(256, n_boxes['conv11_2'] * self.n_points * 2, kernel_size=3, padding=1)

        # final offsets to refine the initial reppoints
        self.loc_conv4_3_out = DeformConv2d(n_boxes['conv4_3'] * self.n_points * 4,
                                            n_boxes['conv4_3'] * self.n_points * 2, kernel_size=1, padding=0)
        self.loc_conv7_out = DeformConv2d(n_boxes['conv4_3'] * self.n_points * 4, n_boxes['conv7'] * self.n_points * 2,
                                          kernel_size=1, padding=0)
        self.loc_conv8_2_out = DeformConv2d(n_boxes['conv4_3'] * self.n_points * 4,
                                            n_boxes['conv8_2'] * self.n_points * 2, kernel_size=1, padding=0)
        self.loc_conv9_2_out = DeformConv2d(n_boxes['conv4_3'] * self.n_points * 4,
                                            n_boxes['conv9_2'] * self.n_points * 2, kernel_size=1, padding=0)
        self.loc_conv10_2_out = nn.Conv2d(n_boxes['conv4_3'] * self.n_points * 4,
                                          n_boxes['conv10_2'] * self.n_points * 2, kernel_size=1, padding=0)
        self.loc_conv11_2_out = nn.Conv2d(n_boxes['conv4_3'] * self.n_points * 4,
                                          n_boxes['conv11_2'] * self.n_points * 2, kernel_size=1, padding=0)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv4_3 = DeformConv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = DeformConv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = DeformConv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = DeformConv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        self.cl_conv4_3_out = DeformConv2d(n_boxes['conv4_3'] * self.n_points * 2 + n_boxes['conv4_3'] * n_classes,
                                           n_boxes['conv4_3'] * n_classes, kernel_size=1, padding=0)
        self.cl_conv7_out = DeformConv2d(n_boxes['conv4_3'] * self.n_points * 2 + n_boxes['conv4_3'] * n_classes,
                                         n_boxes['conv7'] * n_classes, kernel_size=1, padding=0)
        self.cl_conv8_2_out = DeformConv2d(n_boxes['conv4_3'] * self.n_points * 2 + n_boxes['conv4_3'] * n_classes,
                                           n_boxes['conv8_2'] * n_classes, kernel_size=1, padding=0)
        self.cl_conv9_2_out = DeformConv2d(n_boxes['conv4_3'] * self.n_points * 2 + n_boxes['conv4_3'] * n_classes,
                                           n_boxes['conv9_2'] * n_classes, kernel_size=1, padding=0)
        self.cl_conv10_2_out = nn.Conv2d(n_boxes['conv4_3'] * self.n_points * 2 + n_boxes['conv4_3'] * n_classes,
                                         n_boxes['conv10_2'] * n_classes, kernel_size=1, padding=0)
        self.cl_conv11_2_out = nn.Conv2d(n_boxes['conv4_3'] * self.n_points * 2 + n_boxes['conv4_3'] * n_classes,
                                         n_boxes['conv11_2'] * n_classes, kernel_size=1, padding=0)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        Forward propagation.

        :param conv4_3_feats: conv4_3 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: conv7 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param conv10_2_feats: conv10_2 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 5685 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = conv4_3_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        # layer conv4_3
        l_conv4_3_init = self.loc_conv4_3_init(conv4_3_feats)  # (N, 3 * 18, 38, 38)
        l_conv4_3_refine = self.loc_conv4_3_refine(conv4_3_feats)
        l_conv4_3_offset = self.loc_conv4_3_out(torch.cat([F.relu(l_conv4_3_init), F.relu(l_conv4_3_refine)], dim=1))
        l_conv4_3_out = l_conv4_3_init + l_conv4_3_offset

        l_conv4_3_init_out = l_conv4_3_init.permute(0, 2, 3,
                                                    1).contiguous()  # (N, 38, 38, 54), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3_init_out = l_conv4_3_init_out.view(batch_size, -1,
                                                     18)  # (N, 4332, 4), there are a total 4332 boxes on this feature map

        l_conv4_3_out = l_conv4_3_out.permute(0, 2, 3, 1).contiguous()
        l_conv4_3_out = l_conv4_3_out.view(batch_size, -1, 18)

        # layer conv7
        l_conv7_init = self.loc_conv7_init(conv7_feats)  # (N, 3 * 18, 19, 19)
        l_conv7_refine = self.loc_conv7_refine(conv7_feats)
        l_conv7_offset = self.loc_conv7_out(torch.cat([F.relu(l_conv7_init), F.relu(l_conv7_refine)], dim=1))
        l_conv7_out = l_conv7_init + l_conv7_offset
        l_conv7_init_out = l_conv7_init.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 54)
        l_conv7_init_out = l_conv7_init_out.view(batch_size, -1,
                                                 18)  # (N, 1083, 18), there are a total 1083 boxes on this feature map

        l_conv7_out = l_conv7_out.permute(0, 2, 3, 1).contiguous()
        l_conv7_out = l_conv7_out.view(batch_size, -1, 18)

        l_conv8_2_init = self.loc_conv8_2_init(conv8_2_feats)  # (N, 36, 10, 10)
        l_conv8_2_refine = self.loc_conv8_2_refine(conv8_2_feats)
        l_conv8_2_offset = self.loc_conv8_2_out(torch.cat([F.relu(l_conv8_2_init), F.relu(l_conv8_2_refine)], dim=1))
        l_conv8_2_out = l_conv8_2_init + l_conv8_2_offset
        l_conv8_2_init_out = l_conv8_2_init.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 36)
        l_conv8_2_init_out = l_conv8_2_init_out.view(batch_size, -1, 18)  # (N, 200, 18)
        l_conv8_2_out = l_conv8_2_out.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 36)
        l_conv8_2_out = l_conv8_2_out.view(batch_size, -1, 18)  # (N, 200, 18)

        l_conv9_2_init = self.loc_conv9_2_init(conv9_2_feats)  # (N, 36, 5, 5)
        l_conv9_2_refine = self.loc_conv9_2_refine(conv9_2_feats)
        l_conv9_2_offset = self.loc_conv9_2_out(torch.cat([F.relu(l_conv9_2_init), F.relu(l_conv9_2_refine)], dim=1))
        l_conv9_2_out = l_conv9_2_init + l_conv9_2_offset
        l_conv9_2_init_out = l_conv9_2_init.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 36)
        l_conv9_2_init_out = l_conv9_2_init_out.view(batch_size, -1, 18)  # (N, 50, 18)
        l_conv9_2_out = l_conv9_2_out.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 36)
        l_conv9_2_out = l_conv9_2_out.view(batch_size, -1, 18)  # (N, 50, 18)

        l_conv10_2_init = self.loc_conv10_2_init(conv10_2_feats)  # (N, 36, 3, 3)
        l_conv10_2_refine = self.loc_conv10_2_refine(conv10_2_feats)
        l_conv10_2_offset = self.loc_conv10_2_out(
            torch.cat([F.relu(l_conv10_2_init), F.relu(l_conv10_2_refine)], dim=1))
        l_conv10_2_out = l_conv10_2_init + l_conv10_2_offset
        l_conv10_2_init_out = l_conv10_2_init.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 36)
        l_conv10_2_init_out = l_conv10_2_init_out.view(batch_size, -1, 18)  # (N, 27, 4)
        l_conv10_2_out = l_conv10_2_out.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2_out = l_conv10_2_out.view(batch_size, -1, 18)  # (N, 18, 18)

        l_conv11_2_init = self.loc_conv11_2_init(conv11_2_feats)  # (N, 36, 1, 1)
        l_conv11_2_refine = self.loc_conv11_2_refine(conv11_2_feats)
        l_conv11_2_offset = self.loc_conv11_2_out(
            torch.cat([F.relu(l_conv11_2_init), F.relu(l_conv11_2_refine)], dim=1))
        l_conv11_2_out = l_conv11_2_init + l_conv11_2_offset
        l_conv11_2_init_out = l_conv11_2_init.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 36)
        l_conv11_2_init_out = l_conv11_2_init_out.view(batch_size, -1, 18)  # (N, 2, 18)
        l_conv11_2_out = l_conv11_2_out.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 36)
        l_conv11_2_out = l_conv11_2_out.view(batch_size, -1, 18)  # (N, 2, 18)

        # Predict classes in localization boxes
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 3 * n_classes, 38, 38)
        c_conv4_3_out = self.cl_conv4_3_out(torch.cat([F.relu(c_conv4_3), F.relu(l_conv4_3_init)], dim=1))
        c_conv4_3_out = c_conv4_3_out.permute(0, 2, 3,
                                              1).contiguous()  # (N, 38, 38, 3 * n_classes), to match prior-box order (after .view())
        c_conv4_3_out = c_conv4_3_out.view(batch_size, -1,
                                           self.n_classes)  # (N, 4332, n_classes), there are a total 4332 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 3 * n_classes, 19, 19)
        c_conv7_out = self.cl_conv7_out(torch.cat([F.relu(c_conv7), F.relu(l_conv7_init)], dim=1))
        c_conv7_out = c_conv7_out.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 3 * n_classes)
        c_conv7_out = c_conv7_out.view(batch_size, -1,
                                       self.n_classes)  # (N, 1083, n_classes), there are a total 1083 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 2 * n_classes, 10, 10)
        c_conv8_2_out = self.cl_conv8_2_out(torch.cat([F.relu(c_conv8_2), F.relu(l_conv8_2_init)], dim=1))
        c_conv8_2_out = c_conv8_2_out.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 2 * n_classes)
        c_conv8_2_out = c_conv8_2_out.view(batch_size, -1, self.n_classes)  # (N, 200, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 2 * n_classes, 5, 5)
        c_conv9_2_out = self.cl_conv9_2_out(torch.cat([F.relu(c_conv9_2), F.relu(l_conv9_2_init)], dim=1))
        c_conv9_2_out = c_conv9_2_out.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 2 * n_classes)
        c_conv9_2_out = c_conv9_2_out.view(batch_size, -1, self.n_classes)  # (N, 50, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 2 * n_classes, 3, 3)
        c_conv10_2_out = self.cl_conv10_2_out(torch.cat([F.relu(c_conv10_2), F.relu(l_conv10_2_init)], dim=1))
        c_conv10_2_out = c_conv10_2_out.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 2 * n_classes)
        c_conv10_2_out = c_conv10_2_out.view(batch_size, -1, self.n_classes)  # (N, 18, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 2 * n_classes, 1, 1)
        c_conv11_2_out = self.cl_conv11_2_out(torch.cat([F.relu(c_conv11_2), F.relu(l_conv11_2_init)], dim=1))
        c_conv11_2_out = c_conv11_2_out.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 2 * n_classes)
        c_conv11_2_out = c_conv11_2_out.view(batch_size, -1, self.n_classes)  # (N, 2, n_classes)

        # A total of 4332 + 1083 + 200 + 50 + 18 + 2 = 5685 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs_init = torch.cat([l_conv4_3_init_out, l_conv7_init_out, l_conv8_2_init_out,
                               l_conv9_2_init_out, l_conv10_2_init_out, l_conv11_2_init_out], dim=1)  # (N, 5685, 18)
        locs_refine = torch.cat([l_conv4_3_out, l_conv7_out, l_conv8_2_out, l_conv9_2_out,
                                 l_conv10_2_out, l_conv11_2_out], dim=1)  # (N, 5685, 18)
        classes_scores = torch.cat([c_conv4_3_out, c_conv7_out, c_conv8_2_out, c_conv9_2_out,
                                    c_conv10_2_out, c_conv11_2_out], dim=1)  # (N, 5685, n_classes)

        return locs_init, locs_refine, classes_scores


class SSD300RepPoint(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    Default 9 RepPoints at each location
    """

    def __init__(self, n_classes, n_points=9, center_init=True, transform_method='moment'):
        super(SSD300RepPoint, self).__init__()

        self.n_classes = n_classes
        self.n_points = n_points
        self.center_init = center_init
        self.gradient_multipler = 0.1
        self.center_init = center_init

        self.moment_multipler = 0.01
        self.transform_method = transform_method

        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(data=torch.zeros(2), requires_grad=True)

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # Since lower level features (conv4_3_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20.)

        # Prior boxes
        self.rep_points_xy = self.create_rep_points()
        self.priors_xy = self.rep2bbox(self.rep_points_xy)

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 5685 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # Rescale conv4_3 after L2 norm
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)

        # Run auxiliary convolutions (higher level feature map generators)
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)  # (N, 5685, 4), (N, 5685, n_classes)

        return locs, classes_scores

    def create_rep_points(self):
        """
        Create the 5685 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (5685, 18)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': [0.05, 0.1, 0.15],
                      'conv7': [0.2, 0.25, 0.3],
                      'conv8_2': [0.35, 0.4],
                      'conv9_2': [0.5, 0.6],
                      'conv10_2': [0.7, 0.8],
                      'conv11_2': [0.85, 0.9]}

        # aspect_ratios = {'conv4_3': [1., 2., 0.5],
        #                  'conv7': [1., 2., 3., 0.5, .333],
        #                  'conv8_2': [1., 2., 3., 0.5, .333],
        #                  'conv9_2': [1., 2., 3., 0.5, .333],
        #                  'conv10_2': [1., 2., 0.5],
        #                  'conv11_2': [1., 2., 0.5]}

        n_boxes = {'conv4_3': 3,  # the number should be consistent with object scales
                   'conv7': 3,
                   'conv8_2': 2,
                   'conv9_2': 2,
                   'conv10_2': 2,
                   'conv11_2': 2}

        fmaps = list(fmap_dims.keys())

        rep_point_sets = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]  # sliding center locations across the feature maps
                    cy = (i + 0.5) / fmap_dims[fmap]

                    # initalize all reppoints at the center
                    if self.center_init:
                        points = list()
                        for _ in range(self.n_points):
                            points.append(cx)
                            points.append(cy)

                        for _ in range(n_boxes[fmap]):
                            rep_point_sets.append(points)

                    else:
                        # initialize a grid topology of reppoints
                        n_point_side = int(sqrt(self.n_points))
                        for s in range(len(obj_scales[fmap])):
                            scale = obj_scales[fmap][s]
                            interval = torch.linspace(0., scale, n_point_side)
                            points = list()
                            for p in range(self.n_points):
                                for q in range(self.n_points):
                                    points.append(cx - 0.5 * scale + p * interval)
                                    points.append(cy - 0.5 * scale + q * interval)
                            rep_point_sets.append(points)

        prior_points = torch.FloatTensor(rep_point_sets).to(device)
        prior_points.clamp_(0, 1)  # (5685, 18)

        return prior_points

    def rep2bbox(self, predicted_locs):
        """
        Converting the reppoint sets into bounding boxes
        :param predicted_locs: the input point sets, shape: [batchsize, 5685, 18], the element in each set is a list of 18 elements
        e.g. [x1,y1,x2,y1, ..., x9,y9]
        :return: corresponding bounding boxes [x1,y1,x2,y2]
        """
        points_reshape = predicted_locs.view(predicted_locs.size[0], -1, 2, self.n_points)
        pts_x = points_reshape[:, :, 0, :]
        pts_y = points_reshape[:, :, 1, :]

        if self.transform_method == 'min-max':
            bbox_left = pts_x.min(dim=2, keepdim=True)
            bbox_right = pts_x.max(dim=2, keepdim=True)
            bbox_top = pts_y.min(dim=2, keepdim=True)
            bbox_bottom = pts_y.max(dim=2, keep=True)
            bbox = torch.cat([bbox_left, bbox_top, bbox_right, bbox_bottom], dim=2)
        elif self.transform_method == 'moment':
            pts_x_mean = pts_x.mean(dim=2, keepdim=True)
            pts_y_mean = pts_x.mean(dim=2, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=2, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=2, keepdim=True)
            moment_transfer = self.moment_multipler * self.moment_transfer + \
                              (1 - self.moment_multipler) * self.moment_transfer.detach()
            moment_width = moment_transfer[0]
            moment_height = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width)
            half_height = pts_y_std * torch.exp(moment_height)
            bbox = torch.cat([pts_x_mean - half_width, pts_y_mean - half_height,
                              pts_x_mean + half_width, pts_y_mean + half_height], dim=2)

        else:
            raise NotImplementedError

        return bbox.clamp_(0, 1)


    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 5685 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 5685 prior boxes, a tensor of dimensions (N, 5685, 4), with xy representation
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 5685, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        # print('In detect_objects: ')
        batch_size = predicted_locs.size(0)
        n_priors = self.rep_points_xy.size(0)
        # print(n_priors, predicted_locs.size(), predicted_scores.size())
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 5685, n_classes)
        decoded_locs = self.rep2bbox(predicted_locs)  # convert reppoints to bounding boxes

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (5685)

                score_above_min_score = (class_scores > min_score) * 1  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = torch.sum(score_above_min_score).item()

                if n_above_min_score == 0:
                    continue

                class_scores = class_scores[torch.nonzero(score_above_min_score)].squeeze(
                    dim=1)  # (n_qualified), n_min_score <= 5685


                class_decoded_locs = decoded_locs[torch.nonzero(score_above_min_score)].squeeze(
                    dim=1)  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                # suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)
                suppress = torch.zeros((n_above_min_score), dtype=torch.long).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, (overlap[box] > max_overlap) * 1)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                image_boxes.append(class_decoded_locs[torch.nonzero(1 - suppress).squeeze(dim=1)])

                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[torch.nonzero(1 - suppress).squeeze(dim=1)])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class RepPointLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_xy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(RepPointLoss, self).__init__()
        self.priors_xy = priors_xy
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def increase_threshold(self, increment=0.1):
        if self.threshold >= 0.7:
            return

        self.threshold += increment

    def rep2bbox(self, predicted_locs):
        """
        Converting the reppoint sets into bounding boxes
        :param predicted_locs: the input point sets, shape: [batchsize, 5685, 18], the element in each set is a list of 18 elements
        e.g. [x1,y1,x2,y1, ..., x9,y9]
        :return: corresponding bounding boxes [x1,y1,x2,y2]
        """
        points_reshape = predicted_locs.view(predicted_locs.size[0], -1, 2, self.n_points)
        pts_x = points_reshape[:, :, 0, :]
        pts_y = points_reshape[:, :, 1, :]

        if self.transform_method == 'min-max':
            bbox_left = pts_x.min(dim=2, keepdim=True)
            bbox_right = pts_x.max(dim=2, keepdim=True)
            bbox_top = pts_y.min(dim=2, keepdim=True)
            bbox_bottom = pts_y.max(dim=2, keep=True)
            bbox = torch.cat([bbox_left, bbox_top, bbox_right, bbox_bottom], dim=2)
        elif self.transform_method == 'moment':
            pts_x_mean = pts_x.mean(dim=2, keepdim=True)
            pts_y_mean = pts_x.mean(dim=2, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=2, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=2, keepdim=True)
            moment_transfer = self.moment_multipler * self.moment_transfer + \
                              (1 - self.moment_multipler) * self.moment_transfer.detach()
            moment_width = moment_transfer[0]
            moment_height = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width)
            half_height = pts_y_std * torch.exp(moment_height)
            bbox = torch.cat([pts_x_mean - half_width, pts_y_mean - half_height,
                              pts_x_mean + half_width, pts_y_mean + half_height], dim=2)

        else:
            raise NotImplementedError

        return bbox.clamp_(0, 1)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 5685 prior boxes, a tensor of dimensions (N, 5685, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 5685, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 5685, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 5685)

        predicted_bbox = self.rep2bbox(predicted_locs)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 5685)

            # For each prior, find the object that has the maximum overlap, return [value, indices]
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (5685)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (5685), labels[i] is (n_object)

            # print(label_for_each_prior.size(), labels[i].size())
            # exit()

            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (5685)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (5685, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 5685)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_bbox[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 5685)
        # So, if predicted_locs has the shape (N, 5685, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 5685)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 5685)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 5685)
        conf_loss_neg[positive_priors] = 0.  # (N, 5685), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 5685), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 5685)

        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 5685)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
