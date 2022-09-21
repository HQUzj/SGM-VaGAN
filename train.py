import torchvision.transforms as transforms

from utils.eval import *

from vagan import *
import vagan

#############################
# Hyperparameters
#############################
seed = 123
lr = 0.001
beta1 = 0.0
beta2 = 0.9
num_workers = 0
data_path = "dataset"

dis_batch_size = 64  # 64
max_epoch = 40
lambda_kld = 1e-3
latent_dim = 128
cont_dim = 16
cont_k = 8192
cont_temp = 0.07
datasets = 'MNSIT'
# multi-scale contrastive setting
layers = ["b1", "final"]

device_ids = [0]

# device = 1

device = torch.device("cuda:0")

name = ("").join(layers)
log_fname = f"logs/{datasets}-{name}"
fid_fname = f"logs/FID_{datasets}-{name}"
viz_dir = f"viz/{datasets}-{name}"
models_dir = f"saved_models/{datasets}-{name}"
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
lambda_cont = 1.0 / len(layers)
fix_seed(random_seed=seed)

#############################
# Make and initialize the Networks
#############################

net = vagan.SNNgenerator().cuda(device)

dual_encoder = DualDiscriminator(cont_dim).cuda(device)
dual_encoder.apply(weights_init)
dual_encoder_M = DualDiscriminator(cont_dim).cuda(device)

for p, p_momentum in zip(dual_encoder.parameters(), dual_encoder_M.parameters()):
    p_momentum.data.copy_(p.data)
    p_momentum.requires_grad = False
gen_avg_param = copy_params(net.fsdecoder)
d_queue, d_queue_ptr = {}, {}
for layer in layers:
    d_queue[layer] = torch.randn(cont_dim, cont_k).cuda(device)
    d_queue[layer] = F.normalize(d_queue[layer], dim=0)
    d_queue_ptr[layer] = torch.zeros(1, dtype=torch.long)

#############################
# Make the optimizers
#############################
opt_encoder0 = torch.optim.Adam(net.fsencoder.parameters(),
                                0.001, )
opt_decoder0 = torch.optim.Adam(net.fsdecoder.parameters(),
                                0.001, )
opt_encoder = torch.optim.Adam(net.fsencoder.parameters(),
                               lr, (beta1, beta2))
opt_decoder = torch.optim.Adam(net.fsdecoder.parameters(),
                               lr, (beta1, beta2))

shared_params = list(dual_encoder.block1.parameters()) + \
                list(dual_encoder.block2.parameters()) + \
                list(dual_encoder.block3.parameters()) + \
                list(dual_encoder.block4.parameters()) + \
                list(dual_encoder.l5.parameters())
opt_shared = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                     shared_params),
                              3 * lr, (beta1, beta2))
opt_disc_head = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        dual_encoder.head_disc.parameters()),
                                 3 * lr, (beta1, beta2))
cont_params = list(dual_encoder.head_b1.parameters()) + \
              list(dual_encoder.head_b2.parameters()) + \
              list(dual_encoder.head_b3.parameters()) + \
              list(dual_encoder.head_b4.parameters())
opt_cont_head = torch.optim.Adam(filter(lambda p: p.requires_grad, cont_params),
                                 3 * lr, (beta1, beta2))

#############################
# Make the dataloaders
#############################
if datasets == 'CIFAR10':
    ds = torchvision.datasets.CIFAR10(data_path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          (0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5)),
                                  ]))
    train_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                           shuffle=True, pin_memory=True, drop_last=True,
                                           num_workers=num_workers)
    ds = torchvision.datasets.CIFAR10(data_path, train=False, download=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          (0.5, 0.5, 0.5),
                                          (0.5, 0.5, 0.5)),
                                  ]))
    test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                          shuffle=True, pin_memory=True, drop_last=True,
                                          num_workers=num_workers)

elif datasets == 'MNIST':
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
            transforms.Resize((32)),
            transforms.ToTensor(),
            SetRange
        ])
    ds = torchvision.datasets.MNIST(data_path, train=True, download=True,
                                          transform=transform_train
                                    )
    train_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                               shuffle=True, pin_memory=True, drop_last=True,
                                               num_workers=0)
    ds = torchvision.datasets.MNIST(data_path, train=False, download=False,
                                    transform=transform_train
                                    )
    test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              num_workers=0)
elif datasets == 'FashionMNIST':
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform_train = transforms.Compose([
            transforms.Resize((32)),
            transforms.ToTensor(),
            SetRange
        ])
    ds = torchvision.datasets.FashionMNIST(data_path, train=True, download=True,
                                          transform=transform_train
                                    )
    train_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                               shuffle=True, pin_memory=True, drop_last=True,
                                               num_workers=0)
    ds = torchvision.datasets.FashionMNIST(data_path, train=False, download=False,
                                    transform=transform_train
                                    )
    test_loader = torch.utils.data.DataLoader(ds, batch_size=dis_batch_size,
                                              shuffle=True, pin_memory=True, drop_last=True,
                                              num_workers=0)



global_steps = 0
best_fid = 50000
import clean_fid


def calc_clean_fid(network, epoch):
    network = network.eval()
    with torch.no_grad():
        num_gen = 5000
        fid_score = clean_fid.get_clean_fid_score(network, 'CIFAR10', 0,
                                                  num_gen)
        return fid_score

