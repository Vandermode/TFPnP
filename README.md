# TFPnP
[Project Page](https://proceedings.mlr.press/v119/wei20b) | [Paper (ICML version)](https://proceedings.mlr.press/v119/wei20b) | [Paper (JMLR version)](http://jmlr.org/papers/v23/20-1297.html) | [Pretrained Model](https://1drv.ms/u/s!AqddfvhavTRijUiNda2MBmaYHJVL?e=VhYojg)

**Tuning-free Plug-and-Play Proximal Algorithm for Inverse Imaging Problems, ICML 2020 ([Award](https://icml.cc/Conferences/2020/Awards) Paper)** 

Kaixuan Wei, Angelica Aviles-Rivero, Jingwei Liang, Ying Fu, Carola-Bibiane Schönlieb, Hua Huang

## :sparkles: News
- 2022-2-04: The official JMLR version is now available [online](http://jmlr.org/papers/v23/20-1297.html).
- 2021-12-01: The comprehensive version of this work was accepted to *Journal of Machine Learning Research* (JMLR)!
- 2021-11-11: Release the full version with new applications of sparse-view CT (ct) and single-photon imaging (spi). 
- 2021-9-25: Release the initial version. It now includes all sources (code and data) to reproduce our results on the ICML paper. More applications (presented on our journal version) are coming soon. 


## Getting Started

### Install 

```shell
pip install tfpnp
```

**Editable Installation** : Clone the repo, and install the `tfpnp` package first.
For developing purpose, you are recommended to install the package with [```-e```](https://stackoverflow.com/questions/42609943/what-is-the-use-case-for-pip-install-e/59667164#59667164?newreg=9c456c4fac1e46049b0174b263f67d0b) option. 

```shell
git clone https://github.com/Vandermode/TFPnP.git
cd TFPnP
pip install -e .
```



### Testing

1. Download the test data and pretrained models (checkpoints and denoiser) from [Link](https://1drv.ms/u/s!AqddfvhavTRijUiNda2MBmaYHJVL?e=VhYojg), unzip and put them into `tasks/[task]/data`, `tasks/[task]/checkpoints` and `tfpnp/pnp/denoiser/pretrained`
2. Run the test via the following command. (You can find more testing commands in `script.sh` of each task directory)

```shell
cd tasks/csmri
python -W ignore main.py --solver admm --exp test -rs 15000 --max_episode_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_admm_5x6_48/actor_0015000.pkl

cd tasks/pr
python main.py --solver iadmm --exp pr_admm_5x6_36 --eval --max_episode_step 6 --action_pack 5 -r ./checkpoints/pr_admm_5x6_36/actor_0015000.pkl -rs 15000
```

### Training

1. Download the training data from [Link](https://1drv.ms/u/s!AqddfvhavTRijUiNda2MBmaYHJVL?e=VhYojg), unzip and put it in `tasks/[task]/data`.
2. Run the following command to retrain the model. You need about 20G video memory for the training.

```shell
cd tasks/csmri
python -W ignore main.py --solver admm --exp csmri_admm_5x6_48_new --validate_interval 10 --env_batch 48 --rmsize 480 --warmup 20 -lp 0.05 --train_steps 15000 --max_episode_step 6 --action_pack 5 -le 0.2

cd tasks/pr
python main.py --solver iadmm --exp pr_admm_5x6_36 --validate_interval 50 --env_batch 36 --rmsize 360 --warmup 20 -lp 0.05 --train_steps 15000 --max_episode_step 6 --action_pack 5 -le 0.2
```

## Requirement

- The example tasks requires `PyTorch <= 1.7`.
- The package `tfpnp` itself can be used for higher version of PyTorch.


## Citation

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@inproceedings{wei2020tuning,
  title={Tuning-free plug-and-play proximal algorithm for inverse imaging problems},
  author={Wei, Kaixuan and Aviles-Rivero, Angelica and Liang, Jingwei and Fu, Ying and Sch{\"o}nlieb, Carola-Bibiane and Huang, Hua},
  booktitle={International Conference on Machine Learning},
  pages={10158--10169},
  year={2020},
  organization={PMLR}
}

@article{JMLR:v23:20-1297,
  author  = {Kaixuan Wei and Angelica Aviles-Rivero and Jingwei Liang and Ying Fu and Hua Huang and Carola-Bibiane SchÃ¶nlieb},
  title   = {TFPnP: Tuning-free Plug-and-Play Proximal Algorithms with Applications to Inverse Imaging Problems},
  journal = {Journal of Machine Learning Research},
  year    = {2022},
  volume  = {23},
  number  = {16},
  pages   = {1-48},
  url     = {http://jmlr.org/papers/v23/20-1297.html}
}
```

## Contact
If you find any problem, please feel free to contact me (kxwei at princeton.edu ~~kaixuan_wei at bit.edu.cn~~). A brief self-introduction is required, if you would like to get an in-depth help from me.

## Acknowledgments
We thank [@Zeqiang-Lai](https://github.com/Zeqiang-Lai) for code clean and refactoring, which makes it well structured and easy to understand. 
