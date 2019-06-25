import os
import logging
import shutil
import tensorboardX
from losses.sliced_sm import *
from losses.dsm import dsm, select_sigma
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from models.dkef import DKEF, MLP
from losses.score_matching import score_matching, exact_score_matching
import time
import copy
from tqdm import tqdm
import pickle


__all__ = ['DKEFRunner']


class SmallDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class DKEFRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def get_dataset(self):
        # NOTE: in their code, they add the noise to the dataset. (They also have the option to add during
        # train time, but it is not the default).
        if self.config.data.dataset in ["Parkinsons", "RedWine", "WhiteWine"]:
            train_data, val_data, test_data = self.load_and_whiten()
            train_data = torch.tensor(train_data).float()
            train_data = SmallDataset(train_data + torch.randn_like(train_data) * 0.05)
            val_data = torch.tensor(val_data).float()
            val_data = SmallDataset(val_data + torch.randn_like(val_data) * 0.05)
            test_data = torch.tensor(test_data).float()
            test_data = SmallDataset(test_data + torch.randn_like(test_data) * 0.05)
        elif self.config.data.dataset == "HighDim":
            train_data = SmallDataset(np.random.randn(4860, self.args.scalability_dim).astype(np.float32))
            val_data = SmallDataset(np.random.randn(540, self.args.scalability_dim).astype(np.float32))
            test_data = SmallDataset(np.random.randn(600, self.args.scalability_dim).astype(np.float32))
            self.config.data.input_dim = self.args.scalability_dim
        else:
            raise ValueError("Only supports UCI datasets or high dimensional synthetic data")
        return train_data, val_data, test_data

    def clean_data(self, data, cor=0.98):
        C = np.abs(np.corrcoef(data.T))
        B = np.sum(C > cor, axis=1)
        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            data = np.delete(data, col_to_remove, axis=1)
            C = np.corrcoef(data.T)
            B = np.sum(C > cor, axis=1)
        return data

    # NOTE: the below code performs PCA whitening with zero epsilon, as in the DKEF paper.
    def apply_whiten(self, data, compute_W, eps=0):
        if compute_W:
            self.mean = data.mean(0)
            u, s, vt = np.linalg.svd((data - data.mean(0))[:10**4])
            self.W = vt.T / s * np.sqrt(u.shape[0])
            self.Winv = np.linalg.inv(self.W)

        return (data - self.mean) @ self.W

    def inv_whiten(self, data):
        return data @ self.Winv + self.mean

    def dequantize(self, dataset):
        for d in range(dataset.shape[1]):
            diff = np.median(np.diff(np.unique(dataset[:, d])))
            n = dataset.shape[0]
            dataset[:, d] += (np.random.rand(n) * 2 - 1) * diff * 1
        return dataset

    def load_and_whiten(self):
        if self.config.data.dataset == "Parkinsons":
            dataset = np.loadtxt("run/datasets/parkinsons_updrs.data", delimiter=",", skiprows=1)[:,3:]
            dataset = self.clean_data(dataset, cor=0.98)
        elif self.config.data.dataset == "RedWine":
            dataset = np.loadtxt("run/datasets/winequality-red.csv", delimiter=";", skiprows=1)[:,:-1]
            dataset = self.dequantize(dataset)
        elif self.config.data.dataset == "WhiteWine":
            dataset = np.loadtxt("run/datasets/winequality-white.csv", delimiter=";", skiprows=1)[:,:-1]
            dataset = self.dequantize(dataset)

        dataset_size = dataset.shape[0]

        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(dataset)
        np.random.set_state(random_state)

        logging.info("Applying whitening on entire dataset (including test data), as in DKEF paper")
        _ = self.apply_whiten(dataset, compute_W=True)
        test_data_size = min(1000, dataset_size // 10)
        test_data = dataset[-test_data_size:]
        val_data_size = min(1000, (dataset_size - test_data_size) // 10)
        val_data = dataset[-(test_data_size + val_data_size): -test_data_size]
        train_data = dataset[:-(test_data_size + val_data_size)]
        self.config.data.input_dim = dataset.shape[1]
        logging.info("Whitening data")
        train_data = self.apply_whiten(train_data, compute_W=False)
        val_data = self.apply_whiten(val_data, compute_W=False)
        test_data = self.apply_whiten(test_data, compute_W=False)
        logging.info("Whitening completed")

        return train_data, val_data, test_data

    def fisher_divergence(self, energy_net, data, gt_logpdf_net):
        data = data.to(self.config.device)
        data.requires_grad_(True)
        log_pdf_model = -energy_net(data)
        model_score = autograd.grad(log_pdf_model.sum(), data)[0].detach().cpu()
        data = data.cpu()
        data.requires_grad_(True)
        log_pdf_actual = gt_logpdf_net(data)
        actual_score = autograd.grad(log_pdf_actual.sum(), data)[0]
        return 1/2 * ((model_score - actual_score) ** 2).sum(1).mean(0)

    def sample(self, iterator, loader):
        try:
            X = next(iterator)
        except:
            iterator = iter(loader)
            X = next(iterator)
        X = X.to(self.config.device)
        return iterator, X

    def train_stage1(self, dkef, tb_logger, train_data, val_data, collate_fn, train_mode):
        optimizer = self.get_optimizer(dkef.parameters())

        step = 0
        num_mb = len(train_data) // self.config.training.batch_size
        split_size = self.config.training.batch_size // 2
        best_val_step = 0
        best_val_loss = 1e+5
        best_model = None
        train_losses = np.zeros(30)
        val_loss_window = np.zeros(15)
        torch.cuda.synchronize()
        prev_time = time.time()

        val_batch_size = len(val_data)
        num_val_iters = 1
        val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        train_loader = DataLoader(train_data, batch_size=split_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        total_time = 0.0
        secs_per_it = []
        for _ in range(self.config.training.n_epochs):
            for _ in range(num_mb):
                train_iter, X_t = self.sample(train_iter, train_loader)
                train_iter, X_v = self.sample(train_iter, train_loader)

                def energy_net(inputs):
                    return -dkef(X_t, inputs)

                if train_mode == "exact":
                    train_loss = exact_score_matching(energy_net, X_v, train=True)
                elif train_mode == "sliced":
                    train_loss, _, _ = single_sliced_score_matching(energy_net, X_v)
                elif train_mode == "sliced_VR":
                    train_loss, _, _ = sliced_VR_score_matching(energy_net, X_v)
                elif train_mode == "dsm":
                    train_loss = dsm(energy_net, X_v, sigma=self.dsm_sigma)
                elif train_mode == "kingma":
                    logp, grad1, grad2 = dkef.approx_bp_forward(X_t, X_v, stage="train", mode=train_mode)
                    train_loss = (0.5 * grad1 ** 2).sum(1) + grad2.sum(1)
                elif train_mode == "CP":
                    logp, grad1, S_r, S_i = dkef.approx_bp_forward(X_t, X_v, stage="train", mode=train_mode)
                    grad2 = S_r ** 2 - S_i ** 2
                    train_loss = (0.5 * grad1 ** 2).sum(1) + grad2.sum(1)

                train_loss = train_loss.mean()
                optimizer.zero_grad()
                train_loss.backward()
                train_losses[step % 30] = train_loss.detach()

                # Their code clips by overall gradient norm at 100.
                tn = nn.utils.clip_grad_norm_(dkef.parameters(), 100.0)
                optimizer.step()

                idx = np.random.choice(len(train_data), 1000, replace=False)
                train_data_for_val = torch.utils.data.Subset(train_data, idx)
                dkef.save_alpha_matrices(train_data_for_val, collate_fn, self.config.device)

                # Compute validation loss
                def energy_net_val(inputs):
                    return -dkef(None, inputs, stage="eval")

                val_losses = []
                for val_step in range(num_val_iters):
                    val_iter, data_v = self.sample(val_iter, val_loader)
                    if train_mode == "exact":
                        batch_val_loss = exact_score_matching(energy_net_val, data_v, train=False)
                    elif train_mode == "sliced":
                        batch_val_loss, _, _ = single_sliced_score_matching(energy_net_val, data_v, detach=True)
                    elif train_mode == "sliced_VR":
                        batch_val_loss, _, _ = sliced_VR_score_matching(energy_net_val, data_v, detach=True)
                    elif train_mode == "dsm":
                        batch_val_loss = dsm(energy_net_val, data_v, sigma=self.dsm_sigma)
                    elif train_mode == "kingma":
                        logp, grad1, grad2 = dkef.approx_bp_forward(None, X_v, stage="eval", mode=train_mode)
                        batch_val_loss = (0.5 * grad1 ** 2).sum(1) + grad2.sum(1)
                    elif train_mode == "CP":
                        logp, grad1, S_r, S_i = dkef.approx_bp_forward(None, X_v, stage="eval", mode=train_mode)
                        grad2 = S_r ** 2 - S_i ** 2
                        batch_val_loss = (0.5 * grad1 ** 2).sum(1) + grad2.sum(1)

                    val_losses.append(batch_val_loss.mean())

                val_loss = sum(val_losses) / len(val_losses)
                val_loss_window[step % 15] = val_loss.detach()
                smoothed_val_loss = val_loss_window[:step+1].mean() if step < 15 else val_loss_window.mean()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_step = step
                    best_model = copy.deepcopy(dkef.state_dict())
                elif step - best_val_step > self.config.training.patience:
                    self.results["secs_per_it"] = sum(secs_per_it) / len(secs_per_it)
                    self.results["its_per_sec"] = 1. / self.results["secs_per_it"]
                    logging.info("Validation loss has not improved in {} steps. Finalizing model!"
                                 .format(self.config.training.patience))
                    return best_model

                mean_train_loss = train_losses[:step+1].mean() if step < 30 else train_losses.mean()
                logging.info("Step {}, Training loss: {:.2f}, validation loss: {:.2f}".format(step, mean_train_loss, best_val_loss))
                tb_logger.add_scalar('train/train_loss_smoothed', mean_train_loss, global_step=step)
                tb_logger.add_scalar('train/best_val_loss', best_val_loss, global_step=step)
                tb_logger.add_scalar('train/train_loss', train_loss, global_step=step)
                tb_logger.add_scalar('train/val_loss', val_loss, global_step=step)

                if step % 20 == 0:
                    torch.cuda.synchronize()
                    new_time = time.time()
                    logging.info("#" * 80)
                    if step > 0:
                        secs_per_it.append((new_time - prev_time) / 20.)
                    logging.info("Iterations per second: {:.3f}".format(20./(new_time - prev_time)))
                    tb_logger.add_scalar('train/its_per_sec', 20./(new_time - prev_time), global_step=step)

                    if step > 0:
                        total_time += new_time - prev_time

                    val_losses_exact = []
                    for val_step in range(num_val_iters):
                        val_iter, data_v = self.sample(val_iter, val_loader)
                        vle = exact_score_matching(energy_net_val, data_v, train=False)
                        val_losses_exact.append(vle.mean())

                    val_loss_exact = sum(val_losses_exact) / len(val_losses_exact)
                    logging.info("Exact score matching loss on val: {:.2f}".format(val_loss_exact.mean()))
                    tb_logger.add_scalar('eval/exact_score_matching', val_loss_exact.mean(), global_step=step)

                    logging.info("#" * 80)
                    torch.cuda.synchronize()
                    prev_time = time.time()
                step += 1

        logging.info("Completed training")
        self.results["secs_per_it"] = sum(secs_per_it) / len(secs_per_it)
        self.results["its_per_sec"] = 1. / self.results["secs_per_it"]

        return best_model

    def finalize(self, dkef, tb_logger, train_data, val_data, test_data, collate_fn, train_mode):
        lambda_params = [param for (name, param) in dkef.named_parameters() if "lambd" in name]
        optimizer = optim.Adam(lambda_params, lr=0.001)
        batch_size = self.config.training.fval_batch_size
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        dkef.save_alpha_matrices(train_data, collate_fn, self.config.device, override=True)

        def energy_net(inputs):
            return -dkef(None, inputs, stage="finalize")

        step = 0
        while step < 1000:
            for val_batch in val_loader:
                if step >= 1000:
                    break
                val_batch = val_batch.to(self.config.device)

                if train_mode == "exact":
                    val_loss = exact_score_matching(energy_net, val_batch, train=True)
                elif train_mode == "sliced":
                    val_loss, _, _ = single_sliced_score_matching(energy_net, val_batch)
                elif train_mode == "sliced_VR":
                    val_loss, _, _ = sliced_VR_score_matching(energy_net, val_batch)
                elif train_mode == "dsm":
                    val_loss = dsm(energy_net, val_batch, sigma=self.dsm_sigma)
                elif train_mode == "kingma":
                    logp, grad1, grad2 = dkef.approx_bp_forward(None, val_batch, stage="finalize", mode=train_mode)
                    val_loss = (0.5 * grad1 ** 2).sum(1) + grad2.sum(1)
                elif train_mode == "CP":
                    logp, grad1, S_r, S_i = dkef.approx_bp_forward(None, val_batch, stage="finalize", mode=train_mode)
                    grad2 = S_r ** 2 - S_i ** 2
                    val_loss = (0.5 * grad1 ** 2).sum(1) + grad2.sum(1)

                val_loss = val_loss.mean()
                optimizer.zero_grad()
                val_loss.backward()
                optimizer.step()
                logging.info("Val loss: {:.3f}".format(val_loss))
                tb_logger.add_scalar('finalize/loss', val_loss, global_step=step)
                step += 1

        val_losses = []
        for data_v in val_loader:
            data_v = data_v.to(self.config.device)
            batch_val_loss = exact_score_matching(energy_net, data_v, train=False)
            val_losses.append(batch_val_loss.mean())
        val_loss = sum(val_losses) / len(val_losses)
        logging.info("Overall val exact score matching: {:.3f}".format(val_loss))
        tb_logger.add_scalar('finalize/final_valid_score', val_loss, global_step=0)
        self.results["final_valid_score"] = np.asscalar(val_loss.cpu().numpy())

        test_losses = []
        for data_t in test_loader:
            data_t = data_t.to(self.config.device)
            batch_test_loss = exact_score_matching(energy_net, data_t, train=False)
            test_losses.append(batch_test_loss.mean())
        test_loss = sum(test_losses) / len(test_losses)
        logging.info("Overall test exact score matching: {:.3f}".format(test_loss))
        tb_logger.add_scalar('finalize/final_test_score', test_loss, global_step=0)
        self.results["final_test_score"] = np.asscalar(test_loss.cpu().numpy())

    def eval(self, dkef, tb_logger, train_data, val_data, test_data, collate_fn, train_mode):
        q = torch.distributions.MultivariateNormal(torch.zeros(train_data.data.shape[1]), torch.eye(train_data.data.shape[1]) * 4.)

        est = -np.inf
        nsamples = 0
        with torch.no_grad():
            for i in tqdm(range(1000)):
                nsamples += 1000
                samples = q.sample([1000])
                q_prob = q.log_prob(samples)
                samples = samples.to(self.config.device)
                model_prob = dkef(None, samples, stage="finalize")
                diff = model_prob - q_prob.to(self.config.device)
                reduced = torch.logsumexp(diff, dim=0)
                est = np.logaddexp(est, reduced.cpu().numpy())

            self.logZ = est - np.log(nsamples)
            logging.info("Log Norm const: {}".format(self.logZ))

            val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False, num_workers=2, collate_fn=collate_fn)
            val_log_probs = dkef(None, next(iter(val_loader)).to(self.config.device),
                                 stage="eval").cpu().numpy() - self.logZ
            logging.info("Val log prob: {}".format(val_log_probs.mean()))
            tb_logger.add_scalar('finalize/final_val_ll', np.mean(val_log_probs), global_step=0)
            self.results["final_val_ll"] = np.mean(val_log_probs)


            test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=2, collate_fn=collate_fn)
            test_log_probs = dkef(None, next(iter(test_loader)).to(self.config.device)
                                  , stage="eval").cpu().numpy() - self.logZ
            logging.info("Test log prob: {}".format(test_log_probs.mean()))
            tb_logger.add_scalar('finalize/final_test_ll', np.mean(test_log_probs), global_step=0)
            self.results["final_test_ll"] = np.mean(test_log_probs)

        if self.config.data.dataset == "HighDim":
            dist = torch.distributions.MultivariateNormal(torch.zeros(train_data.data.shape[1]),
                                                       torch.eye(train_data.data.shape[1]))
            self.results['actual_val_ll'] = np.mean(dist.log_prob(next(iter(val_loader))).numpy())
            self.results['actual_test_ll'] = np.mean(dist.log_prob(next(iter(test_loader))).numpy())
            logging.info("Actual ll on val and test: {}, {}".
                         format(self.results['actual_val_ll'], self.results['actual_test_ll']))

            def gt_logpdf_net(samples):
                return dist.log_prob(samples)

            def energy_net(samples):
                return dkef(None, samples, stage='eval')

            fisher_divergence_val = self.fisher_divergence(energy_net, next(iter(val_loader)), gt_logpdf_net)
            fisher_divergence_test = self.fisher_divergence(energy_net, next(iter(test_loader)), gt_logpdf_net)
            logging.info("Fisher divergence on val and test: {}, {}".format(fisher_divergence_val, fisher_divergence_test))

            self.results['fisher_divergence_val'] = np.asscalar(fisher_divergence_val.numpy())
            self.results['fisher_divergence_test'] = np.asscalar(fisher_divergence_test.numpy())

    def train(self):
        train_data, val_data, test_data = self.get_dataset()
        collate_fn = torch.utils.data.dataloader.default_collate

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        model_path = os.path.join(self.args.run, 'results', self.args.doc)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        os.makedirs(model_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        self.results = {}
        self.results["doc"] = self.args.doc
        self.results["dataset"] = self.config.data.dataset
        self.results["seed"] = self.args.seed
        self.results["algo"] = self.config.training.algo
        if self.config.training.algo == "dsm":
            # Sigma selection heuristic, commented out in favor of testing values in grid search
            # val_loader = DataLoader(val_data, batch_size=100, shuffle=True, num_workers=2,
            #                         collate_fn=collate_fn)
            # train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2,
            #                           collate_fn=collate_fn)
            # select_sigma(iter(train_loader), iter(val_loader), logit_mnist=False)

            self.dsm_sigma = self.args.dsm_sigma
            self.results["dsm_sigma"] = self.dsm_sigma
        else:
            self.dsm_sigma = None

        # Initialize the model
        init_z_subset = torch.utils.data.Subset(train_data, np.random.choice(len(train_data),
                                                                             self.config.model.M, replace=False))
        init_z_loader = DataLoader(init_z_subset, batch_size=len(init_z_subset), collate_fn=collate_fn)
        init_z = next(iter(init_z_loader)).clone()
        dkef = DKEF(self.config.data.input_dim, mode=self.config.training.algo,
                    num_kernels=self.config.model.num_kernels,
                    init_z=init_z, hidden_dim=self.config.model.hidden_dim, add_skip=self.config.model.add_skip,
                    alpha_param=self.config.model.alpha_param, train_Z=self.config.model.train_Z,
                    pretrained_encoder=None, dsm_sigma=self.dsm_sigma).to(self.config.device)

        # Train the model
        state_dict = self.train_stage1(dkef, tb_logger, train_data, val_data,
                                       collate_fn=collate_fn, train_mode=self.config.training.algo)

        # Reload the model (modifiable to load saved model)
        best_dkef = DKEF(self.config.data.input_dim, mode=self.config.training.algo,
                    num_kernels=self.config.model.num_kernels,
                    init_z=init_z, hidden_dim=self.config.model.hidden_dim, add_skip=self.config.model.add_skip,
                    alpha_param=self.config.model.alpha_param, train_Z=self.config.model.train_Z,
                         pretrained_encoder=None, dsm_sigma=self.dsm_sigma).to(self.config.device)
        best_dkef.load_state_dict(state_dict)

        # Finalize (learn hyperparameters using second step) and evaluate
        self.finalize(best_dkef, tb_logger, train_data, val_data, test_data,
                      collate_fn=collate_fn, train_mode=self.config.training.algo)
        self.eval(best_dkef, tb_logger, train_data, val_data, test_data,
                      collate_fn=collate_fn, train_mode=self.config.training.algo)
        torch.save(best_dkef.state_dict(), model_path + "/model.pth")

        logging.info(self.results)
        pickle_out = open(model_path + "/results.pkl", "wb")
        pickle.dump(self.results, pickle_out)
        pickle_out.close()
