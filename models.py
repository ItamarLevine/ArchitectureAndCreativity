import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from sympy.benchmarks.bench_meijerint import alpha
from torch.nn import functional as F
from sklearn.decomposition import PCA
import torchvision.transforms


NUM_CHANNELS = 3

################################################# network class #################################################
class UNet(nn.Module):
    def __init__(self, device, concat=True, base_channels=32, num_blocks=3, padding_mode='zeros', normalization='regular'):
        super(UNet, self).__init__()
        self.device = device
        self.pool_window = 2
        self.stride = 2
        self.num_blocks = num_blocks
        self.padding = 1
        self.concat = concat
        self.base_channels = base_channels
        self.padding_mode = padding_mode
        self.normalization = normalization

        ########## Encoder ##########
        self.encoder = nn.ModuleDict([])
        for b in range(self.num_blocks):
            self.encoder[str(b)] = self.init_encoder_block(b)

        ########## Mid-layers ##########
        mid_block = nn.ModuleList([])
        for l in range(2):
            if l == 0:
                mid_block.append(
                    nn.Conv2d(self.base_channels * (2 ** b), self.base_channels * (2 ** (b + 1)), 3,
                              padding=self.padding, bias=False, padding_mode=self.padding_mode))
            else:
                mid_block.append(
                    nn.Conv2d(self.base_channels * (2 ** (b + 1)), self.base_channels * (2 ** (b + 1)), 3,
                              padding=self.padding, bias=False, padding_mode=self.padding_mode))
            if self.normalization == 'regular':
                mid_block.append(nn.BatchNorm2d(self.base_channels * (2 ** (b + 1))))
            else:
                mid_block.append(BF_batchNorm(self.base_channels * (2 ** (b + 1))))
            mid_block.append(nn.ReLU(inplace=True))

        self.mid_block = nn.Sequential(*mid_block)

        ########## Decoder ##########
        self.decoder = nn.ModuleDict([])
        self.upsample = nn.ModuleDict([])
        for b in range(self.num_blocks-1, -1, -1):
            self.upsample[str(b)], self.decoder[str(b)] = self.init_decoder_block(b)

    def forward(self, x):
        pool = nn.AvgPool2d(kernel_size=self.pool_window, stride=self.stride, padding=int((self.pool_window - 1) / 2))
        ########## Encoder ##########
        unpooled = []
        for b in range(self.num_blocks):
            x_unpooled = self.encoder[str(b)](x)
            x = pool(x_unpooled)
            unpooled.append(x_unpooled)

        ########## Mid-layers ##########
        x = self.mid_block(x)

        ########## Decoder ##########
        for b in range(self.num_blocks-1, -1, -1):
            x = self.upsample[str(b)](x)
            if self.concat:
                x = torch.cat([x, unpooled[b]], dim=1)
            x = self.decoder[str(b)](x)
        return x

    def init_encoder_block(self, b):
        enc_layers = nn.ModuleList([])
        if b == 0:
            enc_layers.append(
                nn.Conv2d(NUM_CHANNELS, self.base_channels, 3, padding=self.padding, bias=False, padding_mode=self.padding_mode))
            enc_layers.append(nn.ReLU(inplace=True))
            for l in range(1, 2):
                enc_layers.append(nn.Conv2d(self.base_channels, self.base_channels, 3, padding=self.padding, bias=False, padding_mode=self.padding_mode))
                if self.normalization == 'regular':
                    enc_layers.append(nn.BatchNorm2d(self.base_channels))
                else:
                    enc_layers.append(BF_batchNorm(self.base_channels))
                enc_layers.append(nn.ReLU(inplace=True))
        else:
            for l in range(2):
                if l == 0:
                    enc_layers.append(
                        nn.Conv2d(self.base_channels * (2 ** (b - 1)), self.base_channels * (2 ** b), 3, padding=self.padding, bias=False, padding_mode=self.padding_mode))
                else:
                    enc_layers.append(
                        nn.Conv2d(self.base_channels * (2 ** b), self.base_channels * (2 ** b), 3, padding=self.padding, bias=False, padding_mode=self.padding_mode))
                if self.normalization == 'regular':
                    enc_layers.append(nn.BatchNorm2d(self.base_channels * (2 ** b)))
                else:
                    enc_layers.append(BF_batchNorm(self.base_channels * (2 ** b)))
                enc_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*enc_layers)

    def init_decoder_block(self, b):
        dec_layers = nn.ModuleList([])
        dec_padding = self.padding
        # initiate the last block:
        if b == 0:
            for l in range(2 - 1):
                if l == 0:
                    upsample = nn.ConvTranspose2d(self.base_channels * 2, self.base_channels, kernel_size=self.stride, stride=self.stride, bias=False)
                    if self.concat:
                        dec_layers.append(nn.Conv2d(self.base_channels * 2, self.base_channels, kernel_size=3, padding=dec_padding, bias=False, padding_mode=self.padding_mode))
                    else:
                        dec_layers.append(nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=dec_padding, bias=False, padding_mode=self.padding_mode))
                else:
                    dec_layers.append(
                        nn.Conv2d(self.base_channels, self.base_channels, 3, padding=dec_padding, bias=False, padding_mode=self.padding_mode))
                if self.normalization == 'regular':
                    dec_layers.append(nn.BatchNorm2d(self.base_channels))
                else:
                    dec_layers.append(BF_batchNorm(self.base_channels))
                dec_layers.append(nn.ReLU(inplace=True))

            dec_layers.append(
                nn.Conv2d(self.base_channels, NUM_CHANNELS, kernel_size=3, padding=dec_padding, bias=False, padding_mode=self.padding_mode))
        else:
            for l in range(2):
                if l == 0:
                    upsample = nn.ConvTranspose2d(self.base_channels * (2 ** (b+1)), self.base_channels * (2 ** b), kernel_size=2, stride=self.stride, bias=False)
                    if self.concat:
                        dec_layers.append(nn.Conv2d(self.base_channels * (2 ** (b+1)), self.base_channels * (2 ** b), kernel_size=3, padding=dec_padding, bias=False, padding_mode=self.padding_mode))
                    else:
                        dec_layers.append(nn.Conv2d(self.base_channels * (2 ** b), self.base_channels * (2 ** b), kernel_size=2, padding=dec_padding, bias=False, padding_mode=self.padding_mode))
                else:
                    dec_layers.append(
                        nn.Conv2d(self.base_channels * (2 ** b), self.base_channels * (2 ** b), 3, padding=dec_padding, bias=False, padding_mode=self.padding_mode))
                if self.normalization == 'regular':
                    dec_layers.append(nn.BatchNorm2d(self.base_channels * (2 ** b)))
                else:
                    dec_layers.append(BF_batchNorm(self.base_channels * (2 ** b)))
                dec_layers.append(nn.ReLU(inplace=True))
        return upsample, nn.Sequential(*dec_layers)


class BF_batchNorm(nn.Module):
    def __init__(self, num_kernels):
        super(BF_batchNorm, self).__init__()
        self.register_buffer("running_sd", torch.ones(1, num_kernels, 1, 1))
        g = (torch.randn((1, num_kernels, 1, 1)) * (2. / 9. / 64.)).clamp_(-0.025, 0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)

    def forward(self, x):
        training_mode = self.training
        sd_x = torch.sqrt(x.var(dim=(0, 2, 3), keepdim=True, unbiased=False) + 1e-05)
        if training_mode:
            x = x / sd_x.expand_as(x)
            with torch.no_grad():
                self.running_sd.copy_((1 - .1) * self.running_sd.data + .1 * sd_x)

            x = x * self.gammas.expand_as(x)

        else:
            x = x / self.running_sd.expand_as(x)
            x = x * self.gammas.expand_as(x)

        return x


class MultiNoiseUnet(nn.Module):
    def __init__(self, name, device, base_channels=32, num_blocks=3, normalization='regular', padding='zeros'):
        super(MultiNoiseUnet, self).__init__()
        self.name = name
        self.device = device
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.normalization = normalization
        self.padding = padding

    def forward(self, x, t):
        unet = UNet(self.device, base_channels=self.base_channels, num_blocks=self.num_blocks, normalization=self.normalization, padding_mode=self.padding).to(self.device)
        unet.load_state_dict(torch.load(f'{self.name}{t}.pth', map_location=self.device, weights_only=True))
        unet.eval()
        return unet(x)


class PolynomialDenoiser(nn.Module):
    def __init__(self, data_set, device, degree, seed=None, num_features=1024):
        super(PolynomialDenoiser, self).__init__()
        self.degree = degree
        self.device = device
        self.data_set = data_set.flatten(1)
        if seed is not None:
            torch.manual_seed(seed)
        self.rand_mat = torch.randn((num_features, self.data_set.shape[1])).to(self.device) / torch.sqrt(torch.tensor(self.data_set.shape[1]).to(self.device))

    def create_h_data(self, latent, normalization_vector):
        x = latent.flatten(1)
        # if normalization_vector is None:
        #     x = x / x.abs().max(dim=1, keepdim=True)[0]
        # else:
        #     x = x / (normalization_vector[:, None])
        # print(x.shape, x.abs().max(dim=1)[0].shape)
        # x = x / x.abs().max(dim=1)[0][:,None]
        x = self.rand_mat @ x.flatten(1).T
        # x = x/100
        if normalization_vector is None:
            x = x / x.abs().max(dim=0, keepdim=True)[0]
            # x = x / 2
        else:
            x = x / (normalization_vector[None, :])
            # if not train:
        # clean_std = self.data_set.std()
        # theoretical_std = ((1-beta_t) * clean_std + beta_t)**0.5
        # x = x / normalization_vector[None, :]
        # x = x.clamp(-1.0, 1.0)
        # x_std = x.std(dim=1, keepdim=True) + 1e-5
        # x = x / x_std
        # x = nn.Tanh()(x)
        x = x ** self.degree
        x = x / self.data_set.flatten(1).std()
        # x = nn.ReLU()(x)
        return x.T

    def forward(self, x, t, normalization_vector=None, latent_normalization=None):
        latent = x
        beta_t = t
        alpha_t = 1 - beta_t
        noisy_data = (alpha_t ** 0.5) * self.data_set + (beta_t ** 0.5) * torch.randn_like(self.data_set).to(self.device)
        noisy_data = self.create_h_data(noisy_data, None)
        noisy_mean = noisy_data.mean(0)
        noisy_data = noisy_data - noisy_mean
        centered_data = self.data_set - self.data_set.mean(0)
        cov_data_h = (centered_data.T @ noisy_data) / (self.data_set.shape[0] - 1)
        cov_h = (noisy_data.T @ noisy_data) / (self.data_set.shape[0] - 1)
        trace = torch.trace(cov_h)
        relative_epsilon = 1e-4 * (trace / cov_h.shape[0] + 1e-8)
        eye = torch.eye(cov_h.shape[0], device=self.device)
        cov_h_stable = cov_h + 1e-5 * eye
        A = torch.linalg.solve(cov_h_stable, cov_data_h.T).T
        b = self.data_set.mean(0) - A @ noisy_mean
        output =  (A @ self.create_h_data(latent, None).T).T + b
        return output.reshape(x.shape)


class PolynomialPatchDenoiser(nn.Module):
    def __init__(self, data_set, device, degree, patch_size, num_features=1024):
        super(PolynomialPatchDenoiser, self).__init__()
        self.degree = degree
        self.device = device
        self.patch_size = patch_size
        self.num_features = num_features
        self.train_set = data_set
        # self.patch_sizes = [ 64, 64, 45, 25, 17, 17, 9, 7, 5, 3 ]
        self.patch_sizes = [ 64, 55, 45, 35, 27, 21, 15, 11, 7, 3 ]
        self.num_fs = [ 1024, 1024, 512, 256, 128, 128, 64, 32, 16, 8 ] # degree 3
        # self.num_fs = [ 32, 32, 64, 64, 128, 128, 256, 256, 512, 512 ][::-1] # degree 3
        self.pointer = 0

    def forward(self, x, t):
        patch_size = self.patch_sizes[self.pointer]
        image_size = self.train_set.shape[-1]
        denoised = torch.zeros(x.shape[0], NUM_CHANNELS, image_size, image_size).to(self.device)
        noisy_data = (1-t)**0.5 * self.train_set + (t**0.5) * torch.randn_like(self.train_set).to(self.device)
        normalization_vector = noisy_data.flatten(1).abs().max(dim=1)[0]
        for i in range(image_size):
            for j in range(image_size):
                start_i, choose_i = self.get_start_choose_index(i)
                start_j, choose_j = self.get_start_choose_index(j)
                patches = self.train_set[:,:,start_i:start_i+patch_size,start_j:start_j+patch_size]
                poly = PolynomialDenoiser(patches, self.device, degree=self.degree, num_features=self.num_fs[self.pointer]).to(self.device)
                denoised_patch = poly(x[:,:,start_i:start_i+patch_size,start_j:start_j+patch_size], t, normalization_vector, x.flatten(1).abs().max(dim=1)[0])
                if image_size == patch_size:
                    self.pointer += 1
                    print(t, patch_size, denoised_patch.abs().max())
                    return denoised_patch
                else:
                    denoised[:, :, i, j] = denoised_patch[:, :, choose_i, choose_j]
        self.pointer += 1
        return denoised

    def get_start_choose_index(self, index):
        image_size = self.train_set.shape[-1]
        patch_size = self.patch_sizes[self.pointer]
        start_i = max(0, index - patch_size // 2)
        choose_i = min(index, patch_size // 2)
        if start_i + patch_size >= image_size:
            start_i = image_size - patch_size
            choose_i = patch_size - (image_size - index)
        return start_i, choose_i


class LinearDenoiser(nn.Module):
    def __init__(self, data_set, dimension, device):
        super(LinearDenoiser).__init__()
        self.lower_dim_data = None
        self.u = None
        self.device = device
        self.dimension = dimension
        self.mu = data_set.sum(dim=0) / len(data_set)
        self.data_set = data_set.reshape(len(data_set), -1)

    def train(self):
        data_set = self.data_set - self.mu.flatten(start_dim=0)
        u, s, v = torch.linalg.svd(data_set, full_matrices=False)
        self.u =v[:self.dimension]
        self.lower_dim_data = self.u @ data_set.T
        return self

    def forward(self,latents: torch.Tensor, timestep: torch.Tensor):
        if len(latents.shape) != 4:
            latents = latents.unsqueeze(0)
        beta_t = torch.Tensor([timestep]).to(self.device)
        alpha_t = 1 - beta_t
        b_size = latents.shape[0]
        expected_mean_t = self.mu.flatten() * (alpha_t ** 0.5)
        centered_latents = latents.reshape(b_size, -1) - expected_mean_t.unsqueeze(0)
        lower_noisy = self.u @ centered_latents.reshape(b_size, -1).swapaxes(0, 1)
        covariance = torch.linalg.inv(torch.cov((alpha_t ** 0.5) * self.lower_dim_data))
        a = covariance + (1 / beta_t) * torch.eye(covariance.shape[0]).to(self.device)
        b = (1 / beta_t) * lower_noisy
        z_hat = torch.linalg.solve(a, b) / torch.sqrt(alpha_t)
        return (self.u.T @ z_hat).swapaxes(0,1).reshape(latents.shape) + self.mu


class Mlp(nn.Module):
    def __init__(self, device, input_shape=(3, 64, 64), hidden_dim=200):
        super(Mlp, self).__init__()
        self.device = device
        self.c, self.h, self.w = input_shape
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]  # Flattened size
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        # Encoder
        x = self.encoder(x)
        # Decoder
        x = self.decoder(x)
        # Reshape to original image dimensions
        x = x.view(x.size(0), self.c, self.h, self.w )
        return x


class MultiNoiseMlp(nn.Module):
    def __init__(self, name, device, input_shape=(3, 64, 64), hidden_dim=200):
        super(MultiNoiseMlp, self).__init__()
        self.name = name
        self.device = device
        self.c, self.h, self.w = input_shape
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]  # Flattened size
        self.hidden_dim = hidden_dim

    def forward(self, x, sig):
        mlp = Mlp(self.device, input_shape=(self.c, self.h, self.w), hidden_dim=self.hidden_dim).to(self.device)
        mlp.load_state_dict(torch.load(f'{self.name}{sig}.pth', weights_only=True, map_location=self.device))
        mlp.eval()
        return mlp(x)


class TwoDMlp(nn.Module):
    def __init__(self, device, input_shape=2, hidden_dim=64, bottle_neck_dim=1):
        super(TwoDMlp, self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.bottle_neck_dim = bottle_neck_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_shape, hidden_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, bottle_neck_dim),
            nn.LeakyReLU(0.2),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(bottle_neck_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, input_shape),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


class MultiNoiseTwoDMlp(nn.Module):
    def __init__(self,name, device, input_shape=2, hidden_dim=64, bottle_neck_dim=1):
        super(MultiNoiseTwoDMlp, self).__init__()
        self.name = name
        self.device = device
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.bottle_neck_dim = bottle_neck_dim

    def forward(self, x, t):
        model = TwoDMlp(self.device, self.input_shape, self.hidden_dim, self.bottle_neck_dim).to(self.device)
        model.load_state_dict(torch.load(f'{self.name}{t}.pth', weights_only=True, map_location=self.device))
        model.eval()
        return model(x)