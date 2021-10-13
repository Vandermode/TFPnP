python -W ignore main.py --eval --solver iadmm --exp ct_admm_5x6_48_30 -r ./checkpoints/ct_admm_5x6_48_30/actor_0015000.pkl -rs 15000 --max_episode_step 6 --action_pack 5 

python -W ignore main.py --solver iadmm --exp debug --validate_interval 10 --env_batch 12 --rmsize 120 --warmup 20 -lp 0.05 --train_steps 15000 --max_episode_step 6 --action_pack 5 -le 0.2
