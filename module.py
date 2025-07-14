import torch
import torch.nn.functional as F

def conv(batch_norm,
         in_channels,
         out_channels,
         kernel_size=3,
         stride=1,
         dilation=1,
         activation='LeakyReLu'):
    # the basic structure of the network.
    if batch_norm:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation,
                            padding=((kernel_size - 1) * dilation) // 2,
                            bias=False), torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            if activation == 'LeakyReLu' else torch.nn.ReLU(inplace=False)
            )
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation,
                            padding=((kernel_size - 1) * dilation) // 2,
                            bias=True),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
            if activation == 'LeakyReLu' else torch.nn.ReLU(inplace=False))


def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(
        1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(
        1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)

    return grid


class FeatureExtractor(torch.nn.Module):
    # extract image feature with pyramid
    #  return pramid level: [6, 5, 4, 3, 2, 1]
    def __init__(self, argv):
        super(FeatureExtractor, self).__init__()
        self.argv = argv

        self.NetFeature = []
        for l, (ch_in,
                ch_out) in enumerate(zip(argv.lv_chs[:-1], argv.lv_chs[1:])):
            layer = torch.nn.Sequential(
                conv(argv.batch_norm, ch_in, ch_out, stride=2),
                conv(argv.batch_norm, ch_out, ch_out))
            self.add_module(f'Feature(Lv{l})', layer)
            self.NetFeature.append(layer)

    # end

    def forward(self, img):
        feature_pyramid = []
        for net in self.NetFeature:
            img = net(img)
            feature_pyramid.append(img)

        return feature_pyramid[::-1]

class FeatureExtractor_last(torch.nn.Module):
    # extract image feature with pyramid
    #  return pramid level: [6, 5, 4, 3, 2, 1]
    def __init__(self, argv):
        super(FeatureExtractor_last, self).__init__()
        self.argv = argv
        self.FeatureLast = torch.nn.Sequential(
            conv(argv.batch_norm, argv.lv_chs[0], argv.lv_chs[0], stride=1),
            conv(argv.batch_norm, argv.lv_chs[0], argv.lv_chs[0]))


    # end

    def forward(self, img):
        return self.FeatureLast(img)

class costvol_layer(torch.nn.Module):
    # the layer to calculate the corrlation volume between two images
    def __init__(self, argv) -> None:
        super(costvol_layer, self).__init__()
        self.search_range = argv.search_range
        self.argv = argv

    def forward(self, first, second):
        """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
        Args:
            first: Level of the feature pyramid of Image1
            second: Warped level of the feature pyramid of image22
            search_range: Search range (maximum displacement)
        """
        padded_lvl = torch.nn.functional.pad(
            second, (self.search_range, self.search_range, self.search_range,
                     self.search_range)).to(self.argv.device)
        _, _, h, w = first.shape
        max_offset = self.search_range * 2 + 1

        cost_vol = []
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                second_slice = padded_lvl[:, :, y:y + h, x:x + w]
                cost = torch.mean(first * second_slice, dim=1, keepdim=True)
                cost_vol.append(cost)
        cost_vol = torch.cat(cost_vol, dim=1).to(self.argv.device)

        return cost_vol


class Warping_layer(torch.nn.Module):
    # the warping layer to wrap image with the flow
    def __init__(self, argv) -> None:
        super(Warping_layer, self).__init__()
        self.argv = argv

    def forward(self, x, flow):
        argv = self.argv

        flow_for_grip = torch.zeros_like(flow)
        flow_for_grip[:, 0, :, :] = flow[:, 0, :, :] / (
            (flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:, 1, :, :] = flow[:, 1, :, :] / (
            (flow.size(2) - 1.0) / 2.0)

        x_shape = x.size()
        # print(x_shape[0])

        tenHorizontal = torch.linspace(-1.0, 1.0, x_shape[3]).view(1, 1, 1, x_shape[3]).expand(x_shape[0], -1, x_shape[2], -1)

        tenVertical = torch.linspace(-1.0, 1.0, x_shape[2]).view(1, 1, x_shape[2], 1).expand(x_shape[0], -1, -1, x_shape[3])

        if self.argv.device == 'cuda':
            grid_x = torch.cat([tenHorizontal, tenVertical], 1).type(x.type()).cuda()

        else:
            grid_x = torch.cat([tenHorizontal, tenVertical], 1)

        grid = (grid_x - flow_for_grip).permute(0, 2, 3, 1)
        x_warp = torch.nn.functional.grid_sample(x,
                                                 grid,
                                                 mode='bilinear',
                                                #  padding_mode='zeros',
                                                padding_mode='border',
                                                 align_corners=True)

        return x_warp

class Upsample(torch.nn.Module):
    def __init__(self, mode, channels=None, fp16=False):
        super(Upsample, self).__init__()
        self.interp = torch.nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = True
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = torch.nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = torch.nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = torch.nn.Identity()
                self.conv = torch.nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
                ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = torch.nn.Identity()
            self.conv = torch.nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = ((x.shape[2]*2), (x.shape[3]*2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


class FlowEstimator(torch.nn.Module):
    # Estimator: combine the costvol, flow, T, ref to get estimated flow
    def __init__(self, argv, in_ch):
        # in_ch: the input channels
        super(FlowEstimator, self).__init__()
        self.argv = argv

        self.NetMain_flow = torch.nn.Sequential(
            conv(argv.batch_norm, in_ch, 64), conv(argv.batch_norm, 64, 128),
            conv(argv.batch_norm, 128, 96), conv(argv.batch_norm, 96, 64),
            conv(argv.batch_norm, 64, 32),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=2,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                            )

    def forward(self, x):
        return self.NetMain_flow(x)


class TEstimator(torch.nn.Module):
    # Estimator: combine the costvol, flow, T, ref to get estimated T
    def __init__(self, argv, in_ch):
        # in_ch: the input channels
        super(TEstimator, self).__init__()
        self.argv = argv

        self.NetMain_T = torch.nn.Sequential(
            conv(argv.batch_norm, in_ch, 64), conv(argv.batch_norm, 64, 128),
            conv(argv.batch_norm, 128, 96), conv(argv.batch_norm, 96, 64),
            conv(argv.batch_norm, 64, 32),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0),

                            )

    def forward(self, x):
        return self.NetMain_T(x)

class DEstimator(torch.nn.Module):
    # Estimator: combine the costvol, flow, T, ref to get estimated T
    def __init__(self, argv, in_ch):
        # in_ch: the input channels
        super(DEstimator, self).__init__()
        self.argv = argv

        self.NetMain_D = torch.nn.Sequential(
            conv(argv.batch_norm, in_ch, 64), conv(argv.batch_norm, 64, 128),
            conv(argv.batch_norm, 128, 96), conv(argv.batch_norm, 96, 64),
            conv(argv.batch_norm, 64, 32),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                            )

    def forward(self, x):
        return self.NetMain_D(x)


class Refiner_flow(torch.nn.Module):
    # refiner: refine the flow with subpixel resolution
    def __init__(self, argv, ch_feature, upsampling=False):
        #
        # ch_feature: the feature pyramid's channel
        # upsampling: if true, the feature pyramid are upsampled to get finer flow and T
        super(Refiner_flow, self).__init__()
        self.argv = argv
        self.upsampling = upsampling
        self.ch_feature = ch_feature
        if upsampling:
            self.netFeat = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=ch_feature, out_channels=ch_feature*2, kernel_size=1, stride=1, padding=0),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )
            ch_num = 4 * ch_feature + 4
        else:
            self.netFeat = torch.nn.Sequential()
            ch_num = 2 * ch_feature + 4

        self.warping = Warping_layer(self.argv)
        self.phaseC = phaseC_layer(self.argv)
  
        self.NetMain_refiner = torch.nn.Sequential(
            conv(argv.batch_norm, ch_num, 64), conv(argv.batch_norm, 64, 128, 3),
            conv(argv.batch_norm, 128, 96, 3), conv(argv.batch_norm, 96, 64, 3),
            conv(argv.batch_norm, 64, 32, 3),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=2,
                            kernel_size=1,
                            stride=1,
                            padding=0),          
                            )

    def forward(self, imgFeature_first, imgFeature_second, flow, T, D):
        if self.upsampling:
            # upsample to double channel's number
            imgFeature_first = self.netFeat(imgFeature_first)
            imgFeature_second = self.netFeat(imgFeature_second)
        if self.argv.with_T:
            with torch.cuda.amp.autocast(enabled=False):

                imgFeature_first_warp = self.warping(imgFeature_first.float(), flow.float())

                imgFeature_first_warp = T.float() * (torch.mean(imgFeature_first_warp, dim=1, keepdim=True) + D.float() * (imgFeature_first_warp - torch.mean(imgFeature_first_warp, dim=1, keepdim=True))) / self.phaseC(flow.float())

        else:
            with torch.cuda.amp.autocast(enabled=False):
                imgFeature_first = self.warping(imgFeature_first.float(), flow.float()) / self.phaseC(flow.float())

        return flow + self.NetMain_refiner(torch.cat([imgFeature_first, imgFeature_second, flow, T, D], 1))


class Refiner_T(torch.nn.Module):
    # refiner: refine the T with subpixel resolution
    def __init__(self, argv, ch_feature, upsampling=False):
        #
        # ch_feature: the feature pyramid's channel
        # upsampling: if true, the feature pyramid are upsampled to get finer flow and T
        super(Refiner_T, self).__init__()
        self.argv = argv
        self.upsampling = upsampling
        self.ch_feature = ch_feature
        if upsampling:
            self.netFeat = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=ch_feature, out_channels=ch_feature*2, kernel_size=1, stride=1, padding=0),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )

            ch_num = 4 * ch_feature + 4
        else:
            self.netFeat = torch.nn.Sequential()
            ch_num = 2 * ch_feature + 4 + 2

        self.warping = Warping_layer(self.argv)
        self.phaseC = phaseC_layer(self.argv)

        self.NetMain_refiner = torch.nn.Sequential(
            conv(argv.batch_norm, ch_num, 64), conv(argv.batch_norm, 64, 128, 3),
            conv(argv.batch_norm, 128, 96, 3), conv(argv.batch_norm, 96, 64, 3),
            conv(argv.batch_norm, 64, 32, 3),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                            )


    def forward(self, img_first, img_second, flow, T, D):
        if self.upsampling:
            # upsample to double channel's number
            img_first = self.netFeat(img_first)
            img_second = self.netFeat(img_second)

        with torch.cuda.amp.autocast(enabled=False):

            img_first_wrap = self.warping(img_first.float(), flow.float())
            img_first_wrap =  T.float() * (torch.mean(img_first_wrap, dim=1, keepdim=True) + D.float() * (img_first_wrap - torch.mean(img_first_wrap, dim=1, keepdim=True))) / self.phaseC(flow.float())
            
        return T + self.NetMain_refiner(torch.cat([img_first_wrap, img_second, torch.mean(img_second, dim=1, keepdim=True), torch.mean(img_first_wrap, dim=1, keepdim=True), flow, T, D], 1))
        

class Refiner_D(torch.nn.Module):
    # refiner: refine the T with subpixel resolution
    def __init__(self, argv, ch_feature, upsampling=False):
        #
        # ch_feature: the feature pyramid's channel
        # upsampling: if true, the feature pyramid are upsampled to get finer flow and T
        super(Refiner_D, self).__init__()
        self.argv = argv
        self.upsampling = upsampling
        self.ch_feature = ch_feature
        if upsampling:
            self.netFeat = torch.nn.Sequential(
                            torch.nn.Conv2d(in_channels=ch_feature, out_channels=ch_feature*2, kernel_size=1, stride=1, padding=0),
                            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                        )
            ch_num = 4 * ch_feature + 4
        else:
            self.netFeat = torch.nn.Sequential()
            ch_num = 2 * ch_feature + 4 + 2

        self.warping = Warping_layer(self.argv)
        self.phaseC = phaseC_layer(self.argv)

        self.NetMain_refiner = torch.nn.Sequential(
            conv(argv.batch_norm, ch_num, 64), conv(argv.batch_norm, 64, 128, 3),
            conv(argv.batch_norm, 128, 96, 3), conv(argv.batch_norm, 96, 64, 3),
            conv(argv.batch_norm, 64, 32, 3),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=1,
                            kernel_size=1,
                            stride=1,
                            padding=0),
                            )


    def forward(self, img_first, img_second, flow, T, D):
        if self.upsampling:
            # upsample to double channel's number
            img_first = self.netFeat(img_first)
            img_second = self.netFeat(img_second)

        with torch.cuda.amp.autocast(enabled=False):

            img_first_wrap = self.warping(img_first.float(), flow.float())
            img_first_wrap =  T.float() * (torch.mean(img_first_wrap, dim=1, keepdim=True) + D.float() * (img_first_wrap - torch.mean(img_first_wrap, dim=1, keepdim=True))) / self.phaseC(flow.float())

        return D + self.NetMain_refiner(torch.cat([img_first_wrap, img_second, torch.std(img_second, dim=1, unbiased=False, keepdim=True), (torch.std(img_first_wrap, dim=1, unbiased=False, keepdim=True)), flow, T, D], 1))


class phaseC_layer(torch.nn.Module):
    # the calculate phase induced intensity changing
    def __init__(self, argv) -> None:
        super(phaseC_layer, self).__init__()
        self.argv = argv

    def forward(self, flow):

        argv = self.argv
        if argv.phaseC:
            if self.argv.device == 'cuda':

                kernel_x = torch.tensor([[0., 0., 0.],
                                [-0.5, 0., 0.5],
                                [0., 0., 0.]]).type(flow.type()).cuda()
                kernel_y = torch.tensor([[0., -0.5, 0.],
                                    [0., 0., 0.],
                                    [0., 0.5, 0.]]).type(flow.type()).cuda()
                lower_limit = torch.tensor(1e-1).type(flow.type()).cuda()
            else:
                kernel_x = torch.tensor([[0., 0., 0.],
                                        [-0.5, 0., 0.5],
                                        [0., 0., 0.]])
                kernel_y = torch.tensor([[0., -0.5, 0.],
                                        [0., 0., 0.],
                                        [0., 0.5, 0.]])
                lower_limit = torch.tensor(1e-1)

            kernel_x = kernel_x.view(1, 1, 3, 3)
            kernel_y = kernel_y.view(1, 1, 3, 3)

            phase_c = 1.0 + F.conv2d(input=flow[:, 0:1, :, :], weight=kernel_x, padding=1) +\
                        F.conv2d(input=flow[:, 1:2, :, :], weight=kernel_y, padding=1)
        else:
            if self.argv.device == 'cuda':
                lower_limit = torch.tensor(1e-1).type(flow.type()).cuda()
            else:
                lower_limit = torch.tensor(1e-1)
            phase_c = torch.tensor(1)
        
        return torch.maximum(phase_c, lower_limit)

class localStack_layer(torch.nn.Module):

    def __init__(self, n_range=2) -> None:
        super(localStack_layer, self).__init__()
        self.n_range = n_range

    def forward(self, img):
        """Build local stack information
        Args:
            img: image 1
        """
        stack_img = []
        for i in range(-self.n_range, self.n_range+1):
            for j in range(-self.n_range, self.n_range+1):
                stack_img.append(torch.roll(img, shifts=(i,j), dims=(-1,-2)))
        
        stack_img = torch.cat(stack_img, dim=1)
        return stack_img