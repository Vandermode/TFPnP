import os
import time
import torch
from os.path import join

from ..utils.visualize import save_img, seq_plot
from ..utils.metric import psnr_qrnn3d
from ..utils.misc import MetricTracker
from ..utils.log import COLOR, Logger
from ..env.base import PnPEnv


class Evaluator(object):
    def __init__(self, env: PnPEnv, val_loaders, savedir=None, metric=psnr_qrnn3d):
        self.env = env
        self.val_loaders = val_loaders
        self.savedir = savedir
        self.metric = metric
        self.logger = Logger(savedir)

    @torch.no_grad()
    def eval(self, policy, step):
        policy.eval()
        total_metric = 0
        for name, val_loader in self.val_loaders.items():
            metric_tracker = MetricTracker()
            for index, data in enumerate(val_loader):
                assert data['gt'].shape[0] == 1

                # obtain sample's name
                if name in data.keys():
                    data_name = data['name'][0]
                    data.pop('name')
                else:
                    data_name = 'case' + str(index)

                # run
                psnr_init, psnr_finished, info, imgs = eval_single(self.env, data, policy,
                                                                   max_episode_step=self.env.max_episode_step,
                                                                   metric=self.metric)

                episode_steps, psnr_seq, action_seqs, run_time = info
                input, output_init, output, gt = imgs

                # save metric
                metric_tracker.update({'iters': episode_steps,
                                       'psnr_init': psnr_init, 'psnr': psnr_finished, 'time': run_time})

                # save imgs
                if self.savedir is not None:
                    base_dir = join(self.savedir, name, data_name, str(step))
                    os.makedirs(base_dir, exist_ok=True)

                    save_img(input, join(base_dir, 'input.png'))
                    save_img(output_init, join(base_dir, 'output_init.png'))
                    save_img(output, join(base_dir, f'output_{psnr_finished: .2f}.png'))
                    save_img(gt, join(base_dir, 'gt.png'))

                    params = {}
                    for k, v in action_seqs.items():
                        seq_plot(v, 'step', k, save_path=join(base_dir, str(k)+'.png'))
                        params[str(k)] = [float(x) for x in v]
                    
                    import json
                    json.dump(params, open(join(base_dir, 'action_seqs.json'), 'w'))

                    seq_plot(psnr_seq, 'step', 'psnr',
                             save_path=join(base_dir, 'psnr.png'))

            total_metric += metric_tracker['psnr']
            self.logger.log('Step_{:07d}: {} | {}'.format(step - 1, name, metric_tracker), color=COLOR.RED)
        return total_metric / len(self.val_loaders)


def eval_single(env, data, policy, max_episode_step, metric):
    observation = env.reset(data=data)
    hidden = policy.init_state(observation.shape[0])  # TODO: add RNN support
    _, output_init, gt = env.get_images(observation)

    psnr_init = metric(output_init[0], gt[0])

    episode_steps = 0

    psnr_seq = [psnr_init]
    action_seqs = {}

    ob = observation
    time_stamp = time.time()
    while episode_steps < max_episode_step:
        action, _, _, hidden = policy(env.get_policy_ob(ob), idx_stop=None, train=False, hidden=hidden)

        # since batch size = 1, ob and ob_masked are always identicial
        ob, _, _, done, _ = env.step(action)

        episode_steps += 1

        _, output, gt = env.get_images(ob)
        cur_psnr = metric(output[0], gt[0])
        psnr_seq.append(cur_psnr.item())

        action.pop('idx_stop')
        for k, v in action.items():
            if k not in action_seqs.keys():
                action_seqs[k] = []
            for i in range(v.shape[0]):
                action_seqs[k] += list(v[i].detach().cpu().numpy())
        if done:
            break

    run_time = time.time() - time_stamp
    input, output, gt = env.get_images(ob)
    psnr_finished = metric(output[0], gt[0])

    info = (episode_steps, psnr_seq, action_seqs, run_time)
    imgs = (input[0], output_init[0], output[0], gt[0])

    return psnr_init, psnr_finished, info, imgs
