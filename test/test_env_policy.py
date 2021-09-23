import os
import torch.utils.data
from scipy.io import loadmat

from tfpnp.env import CSMRIEnv
from tfpnp.data.dataset import CSMRIDataset, CSMRIEvalDataset
from tfpnp.data.noise_models import GaussianModelD
from tfpnp.pnp.solver.csmri import ADMMSolver_CSMRI
from tfpnp.pnp.denoiser import UNetDenoiser2D
from tfpnp.policy.resnet import ResNetActor
from tfpnp.trainer.a2cddpg.critic import ResNet_wobn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_csmri_loader():
    train_dir = 'data/'
    mri_dir = 'data/'
    mask_dir = 'data/masks'
    
    sigma_ns = [5, 10, 15]
    noise_model = GaussianModelD(sigma_ns)

    sampling_masks = ['radial_128_2', 'radial_128_4', 'radial_128_8']  # different masks
    train_root = os.path.join(train_dir, 'Medical_128')
    val_roots = [os.path.join(mri_dir, 'Medical7_2020', sampling_mask, '15') for sampling_mask in sampling_masks]
    obs_masks = [loadmat(os.path.join(mask_dir, '{}.mat'.format(sampling_mask))).get('mask') for sampling_mask in sampling_masks]
    
    train_dataset = CSMRIDataset(train_root, fns=None, masks=obs_masks, noise_model=noise_model, repeat=12*100)
    val_datasets = [CSMRIEvalDataset(val_root, fns=None) for val_root in val_roots]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)

    val_loaders = [torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True) for val_dataset in val_datasets]
    
    return train_loader, val_loaders


def test_csmri_env():
    train_loader, _ = get_csmri_loader()
    denoiser = UNetDenoiser2D('model/unet-nm.pt').to(device)
    solver = ADMMSolver_CSMRI(denoiser)
    env = CSMRIEnv(train_loader, solver, max_episode_step=6)
    state = env.reset()
    for k,v in state.items():
        try:
            print(k, v.shape)
        except:
            print(k, type(v))

    policy_state = env.get_policy_state(state)
    policy_network = ResNetActor(6+3, 5).to(device)
    action, action_log_prob, dist_entroy = policy_network(policy_state)
    print(action)
    print(action_log_prob)
    print(dist_entroy)
    
    critic = ResNet_wobn(9, 18, 1) .to(device)
    eval_state = env.get_eval_state(state)
    print(eval_state.shape)
    v = critic(eval_state)
    print(v.shape)
    
if __name__ == '__main__':
    test_csmri_env()