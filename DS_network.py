"""
    model of deep speckle. including the follow parts:
    1. feature extractor
    2. flow estimator
    3. intensity estimator
    4. regularizer for subpixel
    5. refiner for flow and intensity
"""

from sys import argv
import torch

from module import FeatureExtractor, costvol_layer, Warping_layer, FlowEstimator, TEstimator, DEstimator, Refiner_flow, Refiner_T, Refiner_D, FeatureExtractor_last, Upsample, phaseC_layer, localStack_layer


class Network(torch.nn.Module):
    def __init__(self, argv):
        super(Network, self).__init__()
        self.argv = argv

        self.netFeatures = FeatureExtractor(self.argv)
        self.netFeaturesLast = FeatureExtractor_last(self.argv)
        self.phaseC = phaseC_layer(self.argv)
        self.local_stack = localStack_layer(n_range=2)
        self.warping = Warping_layer(self.argv)
        
        self.Upsample = Upsample(mode=argv.upsample_mode, channels=2, fp16=argv.fp16)

        if argv.corr == 'costvol_layer':
            self.corr = costvol_layer(argv)

        self.flow_estimators = torch.nn.ModuleList()
        
        for l, ch in enumerate(argv.lv_chs[::-1]):
            
            layer = FlowEstimator(argv, (argv.search_range * 2 + 1)**2 + 2)
            
            self.flow_estimators.append(layer)

        if self.argv.with_T:
            
            self.t_estimator = TEstimator(argv, 2)
            
            self.d_estimator = DEstimator(argv, 2)
           
        self.flow_refiner = Refiner_flow(argv, argv.lv_chs[0], False)

        if self.argv.with_T:
            # refiner for T
            self.t_refiner = Refiner_T(argv, argv.lv_chs[0], False)

            self.d_refiner = Refiner_D(argv, argv.lv_chs[0], False)

        # init
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.bias is not None: torch.nn.init.uniform_(m.bias)
                torch.nn.init.xavier_uniform_(m.weight)

            if isinstance(m, torch.nn.ConvTranspose2d):
                if m.bias is not None: torch.nn.init.uniform_(m.bias)
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, imgs):
        img_first = imgs[:, 0:20, :, :]
        img_second = imgs[:, 20:, :, :]

        imgFeature_first = self.netFeatures(img_first) + [self.netFeaturesLast(img_first)]
        imgFeature_second = self.netFeatures(img_second) + [self.netFeaturesLast(img_second)]

        for lv, (f1, f2) in enumerate(
                zip(imgFeature_first[:self.argv.output_level + 1],
                    imgFeature_second[:self.argv.output_level + 1],
                    )):

            if lv == 0:

                shape = list(f1.size())
                shape[1] = 2
                if self.argv.device == 'cuda':
                    flow = torch.zeros(shape).cuda()
                
                else:
                    flow = torch.zeros(shape)
                   
            else:
                with torch.cuda.amp.autocast(enabled=False):
                    flow = self.Upsample(flow.float()) * 2

            with torch.cuda.amp.autocast(enabled=False):
                f1_warp = self.warping(f1.float(), flow.float())


            # correlation
            corr = self.corr(f2, f1_warp)

            if self.argv.corr_activation:
                corr = torch.nn.functional.leaky_relu_(corr)

            # concat and estimate flow
            
            flow = self.flow_estimators[lv](torch.cat([flow, corr],
                                                                dim=1))
            
        with torch.cuda.amp.autocast(enabled=False):
            
            for i in range(self.argv.num_levels - self.argv.output_level - 1):
                flow = self.Upsample(flow.float()) * 2
            
            img1_warp = self.warping(img_first.float(), flow.float()) 
        
        T = self.t_estimator(torch.cat([torch.mean(img_second, dim=1, keepdim=True), torch.mean(img1_warp, dim=1, keepdim=True)], dim=1))
        D = self.d_estimator(torch.cat([torch.std(img_second, dim=1, keepdim=True), torch.std(img1_warp, dim=1, keepdim=True)*T], dim=1))

        if self.argv.with_refiner and self.argv.output_level ==  self.argv.num_levels - 1:
            flow = self.flow_refiner(imgFeature_first[-1], imgFeature_second[-1], flow, T, D)
            T = self.t_refiner(img_first, img_second, flow, T, D)
            D = self.d_refiner(img_first, img_second, flow, T, D)

        return flow.float(), T.float(), D.float()

