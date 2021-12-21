# Differentiable Annealed Importance Sampling (DAIS)
This repository contains the code to reproduce the DAIS results from the paper [Differentiable Annealed Importance Sampling and the Perils of Gradient Noise](https://openreview.net/forum?id=6rqjgrL7Lq).

## Bayesian Linear Regression

Please first install JAX (https://github.com/google/jax), then reproduce our results by running the following:
```
python blr/dais.py --gamma 0.0
python blr/dais.py --gamma 0.9
python blr.dais.py --bsize 100
```

## Variational Autoencoder
Please first install PyTorch (https://pytorch.org/).

For training, one example command for DAIS with $K = 10$ and $S = 5$ (to adapt annealing scheme, add `--adapt_beta`):
```
python vae/mnist_train.py --lf_step 10 --n_particles 5 --lf_lrate 0.08
```

After training, you can find the saved checkpoint and evaluate it (e.g., using AIS):
```
python vae/mnist_eval.py --ais --n_particles 10 --lf_step 10000 --lf_lrate 0.05 --resume xxx
```

Also, you can reproduce the evaulation results of AIS/HAIS/DAIS by runing `vae/mnist_eval_scaling.py`:
```
python vae/mnist_eval_scaling.py --linear_beta --lf_lrate 0.08 --n_particles 10 --resume xxx
```

## Citation
To cite this work, please use
```
@inproceedings{
	zhang2021differentiable,
	title={Differentiable Annealed Importance Sampling and the Perils of Gradient Noise},
	author={Guodong Zhang and Kyle Hsu and Jianing Li and Chelsea Finn and Roger Baker Grosse},
	booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
	year={2021},
	url={https://openreview.net/forum?id=6rqjgrL7Lq}
}
```
