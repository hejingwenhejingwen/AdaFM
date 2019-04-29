import torch.nn as nn
from . import block as B


class AdaResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, norm_type='batch', act_type='relu',
                 res_scale=1, upsample_mode='upconv', adafm_ksize=1):
        super(AdaResNet, self).__init__()

        norm_layer = B.get_norm_layer(norm_type, adafm_ksize)

        fea_conv = B.conv_block(in_nc, nf, stride=2, kernel_size=3, norm_layer=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_layer=norm_layer, act_type=act_type, res_scale=res_scale)
                         for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_layer=norm_layer, act_type=None)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        upsampler = upsample_block(nf, nf, act_type=act_type)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_layer=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_layer=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),
                                  upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x