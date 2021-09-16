# TFPnP Rewrite

TODO:
- RGB Deblur solver
- psnr range [done]
- out of memory bug [done]
- save options in checkpoints
- incoporate tianshou batch [done]
- idx left bug [done]: rho might be occasionlly to small,which make x NAN after data_solution_admm_sr.
- save model after eval
- save training log
- plot multiple line in one figure
- 区分 state and observation , step用的是state，forward用的ob