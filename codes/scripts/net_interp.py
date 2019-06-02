import torch

coef = 0.8

net_adafm_path = '../../experiments/debug_001_adafmnet_noise75_DIV2K/models/8_G.pth'
net_interp_path = './models/debug_001_adafmnet_noise75_DIV2K/interp_{:.2f}.pth'.format(coef)

net_adafm = torch.load(net_adafm_path)
net_interp = net_adafm.copy()

print('Interpolating with coef = ', coef)

for k, v in net_adafm.items():
    if k.find('transformer') >= 0:
        net_interp[k] = coef * v

torch.save(net_interp, net_interp_path)
