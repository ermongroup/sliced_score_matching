import numpy as np
import torch.nn.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.utils import make_grid, save_image
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from losses.vae import elbo, elbo_ssm, iwae, elbo_kernel
from models.vae import Encoder, Decoder, ImplicitEncoder, MLPDecoder, MLPEncoder, \
    MLPImplicitEncoder, MLPScore, Score
import itertools
from evaluations import fid

__all__ = ['VAERunner']


class VAERunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(0.5, 0.999))
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def train(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.8)], indices[int(num_items * 0.8):]
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)

        elif self.config.data.dataset == 'CELEBA':
            dataset = ImageFolder(root=os.path.join(self.args.run, 'datasets', 'celeba'),
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.7)], indices[
                                                                          int(num_items * 0.7):int(num_items * 0.8)]
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=2)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        decoder = MLPDecoder(self.config).to(self.config.device) if self.config.data.dataset == 'MNIST' \
            else Decoder(self.config).to(self.config.device)
        if self.config.training.algo == 'vae':
            encoder = MLPEncoder(self.config).to(self.config.device) if self.config.data.dataset == 'MNIST' \
                else Encoder(self.config).to(self.config.device)
            optimizer = self.get_optimizer(itertools.chain(encoder.parameters(), decoder.parameters()))
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
                encoder.load_state_dict(states[0])
                decoder.load_state_dict(states[1])
                optimizer.load_state_dict(states[2])
        elif self.config.training.algo == 'ssm':
            score = MLPScore(self.config).to(self.config.device) if self.config.data.dataset == 'MNIST' else \
                Score(self.config).to(self.config.device)
            imp_encoder = MLPImplicitEncoder(self.config).to(self.config.device) if self.config.data.dataset == 'MNIST' \
                else ImplicitEncoder(self.config).to(self.config.device)

            opt_ae = optim.RMSprop(itertools.chain(decoder.parameters(), imp_encoder.parameters()),
                                   lr=self.config.optim.lr)
            opt_score = optim.RMSprop(score.parameters(), lr=self.config.optim.lr)
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
                imp_encoder.load_state_dict(states[0])
                decoder.load_state_dict(states[1])
                score.load_state_dict(states[2])
                opt_ae.load_state_dict(states[3])
                opt_score.load_state_dict(states[4])
        elif self.config.training.algo in ['spectral', 'stein']:
            from models.kernel_score_estimators import SpectralScoreEstimator, SteinScoreEstimator
            imp_encoder = MLPImplicitEncoder(self.config).to(self.config.device) if self.config.data.dataset == 'MNIST' \
                else ImplicitEncoder(self.config).to(self.config.device)
            estimator = SpectralScoreEstimator() if self.config.training.algo == 'spectral' else SteinScoreEstimator()
            optimizer = self.get_optimizer(itertools.chain(imp_encoder.parameters(), decoder.parameters()))
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
                imp_encoder.load_state_dict(states[0])
                decoder.load_state_dict(states[1])
                optimizer.load_state_dict(states[2])

        step = 0
        best_validation_loss = np.inf
        validation_losses = []
        recon_type = 'bernoulli' if self.config.data.dataset == 'MNIST' else 'gaussian'

        for _ in range(self.config.training.n_epochs):
            for _, (X, y) in enumerate(dataloader):
                decoder.train()
                X = X.to(self.config.device)
                if self.config.data.dataset == 'CELEBA':
                    X = X + (torch.rand_like(X) - 0.5) / 128.
                elif self.config.data.dataset == 'MNIST':
                    eps = torch.rand_like(X)
                    X = (eps <= X).float()

                if self.config.training.algo == 'vae':
                    encoder.train()
                    loss, *_ = elbo(encoder, decoder, X, recon_type)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif self.config.training.algo == 'ssm':
                    imp_encoder.train()
                    loss, ssm_loss, *_ = elbo_ssm(imp_encoder, decoder, score, opt_score, X, recon_type,
                                                  training=True, n_particles=self.config.model.n_particles)
                    opt_ae.zero_grad()
                    loss.backward()
                    opt_ae.step()
                elif self.config.training.algo in ['spectral', 'stein']:
                    imp_encoder.train()
                    loss = elbo_kernel(imp_encoder, decoder, estimator, X, recon_type,
                                       n_particles=self.config.model.n_particles)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if step % 10 == 0:
                    try:
                        test_X, _ = next(test_iter)
                    except:
                        test_iter = iter(test_loader)
                        test_X, _ = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    if self.config.data.dataset == 'CELEBA':
                        test_X = test_X + (torch.rand_like(test_X) - 0.5) / 128.
                    elif self.config.data.dataset == 'MNIST':
                        test_eps = torch.rand_like(test_X)
                        test_X = (test_eps <= test_X).float()

                    decoder.eval()
                    if self.config.training.algo == 'vae':
                        encoder.eval()
                        with torch.no_grad():
                            test_loss, *_ = elbo(encoder, decoder, test_X, recon_type)
                            logging.info("loss: {}, test_loss: {}".format(loss.item(), test_loss.item()))
                    elif self.config.training.algo == 'ssm':
                        imp_encoder.eval()
                        test_loss, *_ = elbo_ssm(imp_encoder, decoder, score, None, test_X, recon_type, training=False)
                        logging.info("loss: {}, ssm_loss: {}, test_loss: {}".format(loss.item(), ssm_loss.item(),
                                                                                    test_loss.item()))
                        z = imp_encoder(test_X)
                        tb_logger.add_histogram('z_X', z, global_step=step)
                    elif self.config.training.algo in ['spectral', 'stein']:
                        imp_encoder.eval()
                        with torch.no_grad():
                            test_loss = elbo_kernel(imp_encoder, decoder, estimator, test_X, recon_type, 10)

                            logging.info("loss: {}, test_loss: {}".format(loss.item(), test_loss.item()))

                    validation_losses.append(test_loss.item())
                    tb_logger.add_scalar('loss', loss, global_step=step)
                    tb_logger.add_scalar('test_loss', test_loss, global_step=step)

                    if self.config.training.algo == 'ssm':
                        tb_logger.add_scalar('ssm_loss', ssm_loss, global_step=step)

                if step % 500 == 0:
                    with torch.no_grad():
                        z = torch.randn(100, self.config.model.z_dim, device=X.device)
                        decoder.eval()
                        if self.config.data.dataset == 'CELEBA':
                            samples, _ = decoder(z)
                            samples = samples.view(100, self.config.data.channels, self.config.data.image_size,
                                                   self.config.data.image_size)
                            image_grid = make_grid(samples, 10)
                            image_grid = torch.clamp(image_grid / 2. + 0.5, 0.0, 1.0)
                            data_grid = make_grid(X[:100], 10)
                            data_grid = torch.clamp(data_grid / 2. + 0.5, 0.0, 1.0)
                        elif self.config.data.dataset == 'MNIST':
                            samples_logits = decoder(z)
                            samples = torch.sigmoid(samples_logits)
                            samples = samples.view(100, self.config.data.channels, self.config.data.image_size,
                                                   self.config.data.image_size)
                            image_grid = make_grid(samples, 10)
                            data_grid = make_grid(X[:100], 10)

                        tb_logger.add_image('samples', image_grid, global_step=step)
                        tb_logger.add_image('data', data_grid, global_step=step)

                        if len(validation_losses) != 0:
                            validation_loss = sum(validation_losses) / len(validation_losses)
                            if validation_loss < best_validation_loss:
                                best_validation_loss = validation_loss
                                validation_losses = []
                            # else:
                            #     return 0

                if (step + 1) % 10000 == 0:
                    if self.config.training.algo == 'vae':
                        states = [
                            encoder.state_dict(),
                            decoder.state_dict(),
                            optimizer.state_dict()
                        ]
                    elif self.config.training.algo == 'ssm':
                        states = [
                            imp_encoder.state_dict(),
                            decoder.state_dict(),
                            score.state_dict(),
                            opt_ae.state_dict(),
                            opt_score.state_dict()
                        ]
                    elif self.config.training.algo in ['spectral', 'stein']:
                        states = [
                            imp_encoder.state_dict(),
                            decoder.state_dict(),
                            optimizer.state_dict()
                        ]
                    torch.save(states,
                               os.path.join(self.args.log, 'checkpoint_{}0k.pth'.format((step + 1) // 10000)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

                step += 1
                if step >= self.config.training.n_iters:
                    return 0

    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        decoder = MLPDecoder(self.config).to(self.config.device) if self.config.data.dataset == 'MNIST' \
            else Decoder(self.config).to(self.config.device)
        decoder.eval()
        decoder.load_state_dict(states[1])
        z = torch.randn(100, self.config.model.z_dim, device=self.config.device)
        if self.config.data.dataset == 'CELEBA':
            samples, _ = decoder(z)
            samples = samples.view(100, self.config.data.channels, self.config.data.image_size,
                                   self.config.data.image_size)
            image_grid = make_grid(samples, 10)
            image_grid = torch.clamp(image_grid / 2. + 0.5, 0.0, 1.0)
        elif self.config.data.dataset == 'MNIST':
            samples_logits = decoder(z)
            samples = torch.sigmoid(samples_logits)
            samples = samples.view(100, self.config.data.channels, self.config.data.image_size,
                                   self.config.data.image_size)
            image_grid = make_grid(samples, 10)

        save_image(image_grid, 'image_grid.png')

    def test_fid(self):
        assert self.config.data.dataset == 'CELEBA'
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=False, download=True,
                                 transform=transform)
        elif self.config.data.dataset == 'CELEBA':
            dataset = ImageFolder(root=os.path.join(self.args.run, 'datasets', 'celeba'),
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            test_indices = indices[int(0.8 * num_items):]
            test_dataset = Subset(dataset, test_indices)

        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=2)

        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        get_data_stats = False
        manual = False
        if get_data_stats:
            data_images = []
            for _, (X, y) in enumerate(test_loader):
                X = X.to(self.config.device)
                X = X + (torch.rand_like(X) - 0.5) / 128.
                data_images.extend(X / 2. + 0.5)
                if len(data_images) > 10000:
                    break

            if not os.path.exists(os.path.join(self.args.run, 'datasets', 'celeba140_fid', 'raw_images')):
                os.makedirs(os.path.join(self.args.run, 'datasets', 'celeba140_fid', 'raw_images'))
            logging.info("Saving data images")
            for i, image in enumerate(data_images):
                save_image(image,
                           os.path.join(self.args.run, 'datasets', 'celeba140_fid', 'raw_images', '{}.png'.format(i)))
            logging.info("Images saved. Calculating fid statistics now")
            fid.calculate_data_statics(os.path.join(self.args.run, 'datasets', 'celeba140_fid', 'raw_images'),
                                       os.path.join(self.args.run, 'datasets', 'celeba140_fid'), 50, True, 2048)


        else:
            if manual:
                states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
                decoder = Decoder(self.config).to(self.config.device)
                decoder.eval()
                if self.config.training.algo == 'vae':
                    encoder = Encoder(self.config).to(self.config.device)
                    encoder.load_state_dict(states[0])
                    decoder.load_state_dict(states[1])
                elif self.config.training.algo == 'ssm':
                    score = Score(self.config).to(self.config.device)
                    imp_encoder = ImplicitEncoder(self.config).to(self.config.device)
                    imp_encoder.load_state_dict(states[0])
                    decoder.load_state_dict(states[1])
                    score.load_state_dict(states[2])
                elif self.config.training.algo in ['spectral', 'stein']:
                    from models.kernel_score_estimators import SpectralScoreEstimator, SteinScoreEstimator
                    imp_encoder = ImplicitEncoder(self.config).to(self.config.device)
                    imp_encoder.load_state_dict(states[0])
                    decoder.load_state_dict(states[1])

                all_samples = []
                logging.info("Generating samples")
                for i in range(100):
                    with torch.no_grad():
                        z = torch.randn(100, self.config.model.z_dim, device=self.config.device)
                        samples, _ = decoder(z)
                        samples = samples.view(100, self.config.data.channels, self.config.data.image_size,
                                               self.config.data.image_size)
                        all_samples.extend(samples / 2. + 0.5)

                if not os.path.exists(os.path.join(self.args.log, 'samples', 'raw_images')):
                    os.makedirs(os.path.join(self.args.log, 'samples', 'raw_images'))
                logging.info("Images generated. Saving images")
                for i, image in enumerate(all_samples):
                    save_image(image, os.path.join(self.args.log, 'samples', 'raw_images', '{}.png'.format(i)))
                logging.info("Generating fid statistics")
                fid.calculate_data_statics(os.path.join(self.args.log, 'samples', 'raw_images'),
                                           os.path.join(self.args.log, 'samples'), 50, True, 2048)
                logging.info("Statistics generated.")
            else:
                for iter in range(1, 11):
                    states = torch.load(os.path.join(self.args.log, 'checkpoint_{}0k.pth'.format(iter)),
                                        map_location=self.config.device)
                    decoder = Decoder(self.config).to(self.config.device)
                    decoder.eval()
                    if self.config.training.algo == 'vae':
                        encoder = Encoder(self.config).to(self.config.device)
                        encoder.load_state_dict(states[0])
                        decoder.load_state_dict(states[1])
                    elif self.config.training.algo == 'ssm':
                        score = Score(self.config).to(self.config.device)
                        imp_encoder = ImplicitEncoder(self.config).to(self.config.device)
                        imp_encoder.load_state_dict(states[0])
                        decoder.load_state_dict(states[1])
                        score.load_state_dict(states[2])
                    elif self.config.training.algo in ['spectral', 'stein']:
                        from models.kernel_score_estimators import SpectralScoreEstimator, SteinScoreEstimator
                        imp_encoder = ImplicitEncoder(self.config).to(self.config.device)
                        imp_encoder.load_state_dict(states[0])
                        decoder.load_state_dict(states[1])

                    all_samples = []
                    logging.info("Generating samples")
                    for i in range(100):
                        with torch.no_grad():
                            z = torch.randn(100, self.config.model.z_dim, device=self.config.device)
                            samples, _ = decoder(z)
                            samples = samples.view(100, self.config.data.channels, self.config.data.image_size,
                                                   self.config.data.image_size)
                            all_samples.extend(samples / 2. + 0.5)

                    if not os.path.exists(os.path.join(self.args.log, 'samples', 'raw_images_{}0k'.format(iter))):
                        os.makedirs(os.path.join(self.args.log, 'samples', 'raw_images_{}0k'.format(iter)))
                    else:
                        shutil.rmtree(os.path.join(self.args.log, 'samples', 'raw_images_{}0k'.format(iter)))
                        os.makedirs(os.path.join(self.args.log, 'samples', 'raw_images_{}0k'.format(iter)))

                    if not os.path.exists(os.path.join(self.args.log, 'samples', 'statistics_{}0k'.format(iter))):
                        os.makedirs(os.path.join(self.args.log, 'samples', 'statistics_{}0k'.format(iter)))
                    else:
                        shutil.rmtree(os.path.join(self.args.log, 'samples', 'statistics_{}0k'.format(iter)))
                        os.makedirs(os.path.join(self.args.log, 'samples', 'statistics_{}0k'.format(iter)))

                    logging.info("Images generated. Saving images")
                    for i, image in enumerate(all_samples):
                        save_image(image, os.path.join(self.args.log, 'samples', 'raw_images_{}0k'.format(iter),
                                                       '{}.png'.format(i)))
                    logging.info("Generating fid statistics")
                    fid.calculate_data_statics(os.path.join(self.args.log, 'samples', 'raw_images_{}0k'.format(iter)),
                                               os.path.join(self.args.log, 'samples', 'statistics_{}0k'.format(iter)),
                                               50, True, 2048)
                    logging.info("Statistics generated.")
                    fid_number = fid.calculate_fid_given_paths([
                        'run/datasets/celeba140_fid/celeba_test.npz',
                        os.path.join(self.args.log, 'samples', 'statistics_{}0k'.format(iter), 'celeba_test.npz')]
                        , 50, True, 2048)
                    logging.info("Number of iters: {}0k, FID: {}".format(iter, fid_number))

    def test_ais(self):
        assert self.config.data.dataset == 'MNIST'
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=False, download=True,
                                 transform=transform)
        elif self.config.data.dataset == 'CELEBA':
            dataset = ImageFolder(root=os.path.join(self.args.run, 'datasets', 'celeba'),
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            test_indices = indices[int(0.8 * num_items):]
            test_dataset = Subset(dataset, test_indices)

        # test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=False,
        #                          num_workers=2)

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                 num_workers=2)

        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        decoder = MLPDecoder(self.config).to(self.config.device)
        if self.config.training.algo == 'vae':
            encoder = MLPEncoder(self.config).to(self.config.device)
            encoder.load_state_dict(states[0])
            decoder.load_state_dict(states[1])
        elif self.config.training.algo == 'ssm':
            score = MLPScore(self.config).to(self.config.device)
            imp_encoder = MLPImplicitEncoder(self.config).to(self.config.device)
            imp_encoder.load_state_dict(states[0])
            decoder.load_state_dict(states[1])
            score.load_state_dict(states[2])
        elif self.config.training.algo in ['spectral', 'stein']:
            imp_encoder = MLPImplicitEncoder(self.config).to(self.config.device)
            imp_encoder.load_state_dict(states[0])
            decoder.load_state_dict(states[1])

        recon_type = 'bernoulli' if self.config.data.dataset == 'MNIST' else 'gaussian'

        def recon_energy(X, z):
            if recon_type is 'gaussian':
                mean_x, logstd_x = decoder(z)
                recon = (X - mean_x) ** 2 / (2. * (2 * logstd_x).exp()) + np.log(2. * np.pi) / 2. + logstd_x
                recon = recon.sum(dim=(1, 2, 3))
            elif recon_type is 'bernoulli':
                x_logits = decoder(z)
                recon = F.binary_cross_entropy_with_logits(input=x_logits, target=X, reduction='none')
                recon = recon.sum(dim=[1, 2, 3])
            return recon

        from evaluations.ais import AISLatentVariableModels
        ais = AISLatentVariableModels(recon_energy,
                                      self.config.model.z_dim,
                                      self.config.device, n_Ts=1000)

        total_l = 0.
        total_n = 0
        for _, (X, y) in enumerate(test_loader):
            X = X.to(self.config.device)
            if self.config.data.dataset == 'CELEBA':
                X = X + (torch.rand_like(X) - 0.5) / 128.
            elif self.config.data.dataset == 'MNIST':
                eps = torch.rand_like(X)
                X = (eps <= X).float()

            ais_lb = ais.ais(X).mean().item()
            total_l += ais_lb * X.shape[0]
            total_n += X.shape[0]
            print('current ais lb: {}, mean ais lb: {}'.format(ais_lb, total_l / total_n))

    def test_svi(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=False, download=True,
                                 transform=transform)
        elif self.config.data.dataset == 'CELEBA':
            dataset = ImageFolder(root=os.path.join(self.args.run, 'datasets', 'celeba'),
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            test_indices = indices[int(0.8 * num_items):]
            test_dataset = Subset(dataset, test_indices)

        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=False,
                                 num_workers=2)

        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        decoder = MLPDecoder(self.config).to(self.config.device)
        if self.config.training.algo == 'vae':
            encoder = MLPEncoder(self.config).to(self.config.device)
            encoder.load_state_dict(states[0])
            decoder.load_state_dict(states[1])
        elif self.config.training.algo == 'ssm':
            score = Score(self.config).to(self.config.device)
            imp_encoder = MLPImplicitEncoder(self.config).to(self.config.device)
            imp_encoder.load_state_dict(states[0])
            decoder.load_state_dict(states[1])
            score.load_state_dict(states[2])

        total_l = 0.
        total_n = 0
        recon_type = 'bernoulli' if self.config.data.dataset == 'MNIST' else 'gaussian'
        from models.gmm import Gaussian4SVI
        for batch, (X, y) in enumerate(test_loader):
            X = X.to(self.config.device)
            if self.config.data.dataset == 'CELEBA':
                X = X + (torch.rand_like(X) - 0.5) / 128.
            elif self.config.data.dataset == 'MNIST':
                eps = torch.rand_like(X)
                X = (eps <= X).float()

            gaussian = Gaussian4SVI(X.shape[0], self.config.model.z_dim).to(self.config.device)
            optimizer = optim.SGD(gaussian.parameters(), lr=0.01, momentum=0.5)
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.3)
            for i in range(300):
                lr_scheduler.step()
                loss = iwae(gaussian, decoder, X, type=recon_type, k=10, training=True)
                decoder.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(i, loss.item())
            loss = iwae(gaussian, decoder, X, type=recon_type, k=10, training=False)
            total_l += loss.item() * X.shape[0]
            total_n += X.shape[0]
            print('mini-batch: {}, current iwae-10: {}, average iwae-10: {}'.format(batch + 1, loss.item(),
                                                                                    total_l / total_n))
