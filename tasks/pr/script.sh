# python main.py --solver iadmm --exp pr_admm_5x6_36 --validate_interval 50 --env_batch 36 --rmsize 360 --warmup 20 -lp 0.05 --train_steps 15000 --max_episode_step 6 --action_pack 5 -le 0.2
python main.py --solver iadmm --exp pr_admm_5x6_36 --eval --max_episode_step 6 --action_pack 5 -r ./checkpoints/pr_admm_5x6_36/actor_0015000.pkl -rs 15000

python main.py --solver iadmm --exp debug --validate_interval 50 --env_batch 12 --rmsize 120 --warmup 5 -lp 0.05 --train_steps 15000 --max_episode_step 6 --action_pack 5 -le 0.2 --debug

