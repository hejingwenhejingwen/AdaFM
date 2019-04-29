import torch

alpha = 0.8

net_adafm_path = './models/Adaptive_ResNet.pth'
net_interp_path = './models/interp_{:02d}.pth'.format(int(alpha*10))

net_adafm = torch.load(net_adafm_path)
net_interp = net_adafm.copy()

print('Interpolating with alpha = ', alpha)

for k, v in net_adafm.items():
    if k.find('transformer') >= 0:
        net_interp[k] = alpha * v

torch.save(net_interp, net_interp_path)
