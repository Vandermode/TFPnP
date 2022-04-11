# Test Results

## CSMRI

```shell
(torch) ➜  csmri git:(master) ✗ python -W ignore main.py --solver admm --exp test -rs 15000 --max_episode_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_admm_5x6_48/actor_0015000.pkl
[i] Exp dir: ./checkpoints/test
[i] use denoiser: unet
[i] use solver: admm
 Step_0014999: radial_128_2_15 | acc_reward: 4.18 | iters: 1.00 | psnr: 30.28 | psnr_init: 26.16 | 
 Step_0014999: radial_128_4_15 | acc_reward: 4.27 | iters: 2.00 | psnr: 28.57 | psnr_init: 24.28 | 
 Step_0014999: radial_128_8_15 | acc_reward: 4.36 | iters: 3.00 | psnr: 26.51 | psnr_init: 22.07 | 
(torch) ➜  csmri git:(master) ✗ python -W ignore main.py --solver hqs --exp test -rs 15000 --max_episode_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_hqs_5x6_48/actor_0015000.pkl
[i] Exp dir: ./checkpoints/test
[i] use denoiser: unet
[i] use solver: hqs
 Step_0014999: radial_128_2_15 | acc_reward: 4.06 | iters: 2.00 | psnr: 30.21 | psnr_init: 26.16 | 
 Step_0014999: radial_128_4_15 | acc_reward: 4.07 | iters: 3.00 | psnr: 28.42 | psnr_init: 24.28 | 
 Step_0014999: radial_128_8_15 | acc_reward: 3.73 | iters: 4.43 | psnr: 25.95 | psnr_init: 22.07 | 
(torch) ➜  csmri git:(master) ✗  python -W ignore main.py --solver pg --exp test -rs 15000 --max_episode_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_pg_5x6_48/actor_0015000.pkl
[i] Exp dir: ./checkpoints/test
[i] use denoiser: unet
[i] use solver: pg
 Step_0014999: radial_128_2_15 | acc_reward: 4.18 | iters: 1.00 | psnr: 30.28 | psnr_init: 26.16 | 
 Step_0014999: radial_128_4_15 | acc_reward: 4.24 | iters: 2.14 | psnr: 28.54 | psnr_init: 24.28 | 
 Step_0014999: radial_128_8_15 | acc_reward: 4.07 | iters: 4.14 | psnr: 26.27 | psnr_init: 22.07 | 
(torch) ➜  csmri git:(master) ✗ python -W ignore main.py --solver apg --exp test -rs 15000 --max_episode_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_apg_5x6_48/actor_0015000.pkl
[i] Exp dir: ./checkpoints/test
[i] use denoiser: unet
[i] use solver: apg
 Step_0014999: radial_128_2_15 | acc_reward: 4.19 | iters: 1.00 | psnr: 30.29 | psnr_init: 26.16 | 
 Step_0014999: radial_128_4_15 | acc_reward: 4.25 | iters: 2.00 | psnr: 28.55 | psnr_init: 24.28 | 
 Step_0014999: radial_128_8_15 | acc_reward: 4.25 | iters: 2.00 | psnr: 26.36 | psnr_init: 22.07 | 
(torch) ➜  csmri git:(master) ✗ python -W ignore main.py --solver redadmm --exp test -rs 15000 --max_episode_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_red_5x6_48/actor_0015000.pkl
[i] Exp dir: ./checkpoints/test
[i] use denoiser: unet
[i] use solver: redadmm
 Step_0014999: radial_128_2_15 | acc_reward: 4.13 | iters: 1.71 | psnr: 30.26 | psnr_init: 26.16 | 
 Step_0014999: radial_128_4_15 | acc_reward: 4.21 | iters: 3.14 | psnr: 28.56 | psnr_init: 24.28 | 
 Step_0014999: radial_128_8_15 | acc_reward: 3.88 | iters: 4.43 | psnr: 26.10 | psnr_init: 22.07 | 
 ```

## PR

```shell
(torch) ➜  pr git:(master) ✗ python main.py --solver iadmm --exp pr_admm_5x6_36 --eval --max_episode_step 6 --action_pack 5 -r ./checkpoints/pr_admm_5x6_36/actor_0015000.pkl -rs 15000

[i] Exp dir: ./checkpoints/pr_admm_5x6_36
[i] use denoiser: unet
[i] use solver: iadmm

 Step_0014999: alpha_9 | acc_reward: 37.11 | iters: 3.75 | psnr: 40.94 | psnr_init: 3.69 | 
 Step_0014999: alpha_27 | acc_reward: 30.30 | iters: 2.25 | psnr: 34.06 | psnr_init: 3.69 | 
 Step_0014999: alpha_81 | acc_reward: 24.56 | iters: 2.00 | psnr: 28.31 | psnr_init: 3.69 | 
```

## CT

```shell

```

## SPI

```shell
(torch) ➜  spi git:(master) ✗ python -W ignore main.py --eval --solver admm_spi --exp spi_admm_1x10_48 -r checkpoints/spi_admm_1x10_48/actor_0015000.pkl -rs 15000 --max_episode_step 10 --action_pack 1
[i] Exp dir: log/spi_admm_1x10_48
[i] use denoiser: unet
[i] use solver: admm_spi
Step_0014999: spi_x4 | acc_reward: 9.58 | iters: 9.77 | psnr: 25.55 | psnr_init: 15.54 | time: 0.17 | 
Step_0014999: spi_x6 | acc_reward: 11.21 | iters: 7.23 | psnr: 28.47 | psnr_init: 16.95 | time: 0.10 | 
Step_0014999: spi_x8 | acc_reward: 12.57 | iters: 4.23 | psnr: 30.32 | psnr_init: 17.58 | time: 0.06 | 
```