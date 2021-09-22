# python -W ignore main.py --solver admm --exp test -rs 15000 --max_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_admm_5x6_48/actor_0015000.pkl
# python -W ignore main.py --solver hqs --exp test -rs 15000 --max_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_hqs_5x6_48/actor_0015000.pkl
# python -W ignore main.py --solver pg --exp test -rs 15000 --max_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_pg_5x6_48/actor_0015000.pkl
python -W ignore main.py --solver apg --exp test -rs 15000 --max_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_apg_5x6_48/actor_0015000.pkl
python -W ignore main.py --solver redadmm --exp test -rs 15000 --max_step 6 --action_pack 5 --eval -r ./checkpoints/csmri_red_5x6_48/actor_0015000.pkl
