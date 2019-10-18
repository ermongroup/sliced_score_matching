# Sliced Score Matching: A Scalable Approach to Density and Score Estimation

This repo contains a PyTorch implementation for the paper [Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/abs/1905.07088), UAI 2019. Sliced score matching is a scalable variant of score matching that can be used to train unnormalized statistical models or estimating the score (derivatives of the log-density function) of data.



## Dependencies

The following are packages needed for running this repo.

- PyTorch==1.0.1
- TensorFlow==1.12.0
- tqdm
- tensorboardX
- Scipy
- PyYAML



## Running the experiments
```bash
python main.py --runner [runner name] --config [config file]
```

Here `runner name` is one of the following:

- `DKEFRunner`. This corresponds to experiments on deep kernel exponential families.
- `NICERunner`. This corresponds to the sanity check experiment of training a NICE model.
- `VAERunner`. Experiments on VAEs.
- `WAERunner`. Experiments on Wasserstein Auto-Encoders (WAEs).

and `config file` is the directory of some YAML file in `configs/`.



For example, if you want to train an implicit VAE of latent size 8 on MNIST with Sliced Score Matching, just run

```bash
python main.py --runner VAERunner --config vae/mnist_ssm_8.yml
```



## References

If you find the idea or code useful for your research, please consider citing our paper:

```
@inproceedings{song2019sliced,
  author    = {Yang Song and
               Sahaj Garg and
               Jiaxin Shi and
               Stefano Ermon},
  title     = {Sliced Score Matching: {A} Scalable Approach to Density and Score
               Estimation},
  booktitle = {Proceedings of the Thirty-Fifth Conference on Uncertainty in Artificial
               Intelligence, {UAI} 2019, Tel Aviv, Israel, July 22-25, 2019},
  pages     = {204},
  year      = {2019},
  url       = {http://auai.org/uai2019/proceedings/papers/204.pdf},
}
```

