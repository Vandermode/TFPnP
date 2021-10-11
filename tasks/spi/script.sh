CUDA_VISIBLE_DEVICES=0 python -W ignore main.py --eval --solver admm_spi --exp spi_admm_1x10_48 -r checkpoints/spi_admm_1x10_48/actor_0015000.pkl -rs 15000 --max_episode_step 10 --action_pack 1

CUDA_VISIBLE_DEVICES=0 python -W ignore main.py --eval --solver admm_spi --exp spi_admm_1x20_48 -r checkpoints/spi_admm_1x20_48/actor_0015000.pkl -rs 15000 --max_episode_step 20 --action_pack 1


CUDA_VISIBLE_DEVICES=0 python main.py --solver admm_spi --exp debug --validate_interval 10 --env_batch 12 --rmsize 100 --warmup 20 -lp 0.05 --train_steps 15000 --max_episode_step 20 --action_pack 1 -le 0.2
