import time
# import gymnasium as gym
from multiprocessing import Pool
import core
import torch
from torch.optim import Adam
import numpy as np
import imageio
import wandb
import os
import json
import pathlib
from settings import default_sim_settings, make_cfg
from walker_env_habitat import WalkerEnvHabitat
import habitat_sim.physics as phy
# from pathos.multiprocessing import ProcessingPool as Pool

sim_settings = default_sim_settings
sim_settings["scene"] = "NONE"
sim_settings.pop("scene_dataset_config_file")

def env_fn(bvh_path):
    return WalkerEnvHabitat(sim_settings, bvh_path)

class PPO():
    def __init__(self, env, args, ac_kwargs=dict(), device='cpu'):
        self.num_envs = args.num_envs
        self.env_fn = env
        # self.envs = [env(args.bvh_path) for _ in range(self.num_envs)]
        self.bvh_path = args.bvh_path
        env_ = env(self.bvh_path)
        self.ac = core.MLPActorCritic(env_.get_observation().shape[0], len(env_.agent_model.joint_forces), args.log_policy_var, **ac_kwargs)
        self.show_time = True
        self.episodes_per_epoch = args.episodes_per_epoch
        self.local_episodes_per_epoch = int(self.episodes_per_epoch / self.num_envs)
        self.gamma, self.lam = args.gamma, args.lam
        self.seed = args.seed
        
        self.clip_ratio = args.clip_ratio
        self.target_kl = args.target_kl
        self.kl_multiplier = args.kl_multiplier

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=args.pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=args.vf_lr)

        self.train_pi_iters = args.train_pi_iters
        self.train_v_iters = args.train_v_iters

        self.total_episodes = args.total_episodes
        self.episodes = 0
        self.log_episodes = args.log_episodes
        self.v_steps = 0
        self.pi_steps = 0
        self.exp_name = args.exp_name
        self.args = args

    def _collect_trajectories_sequential(self, env_id):
        # collect trajectories using self.envs[env_id]
        seed = self.seed + 10000 * env_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = env_fn(self.bvh_path)
        trajs = []
        obs_buf = []
        act_buf = []
        rews_buf = []
        adv_buf = []
        ret_buf = []
        val_buf = []
        logp_buf = []

        for i in range(self.local_episodes_per_epoch):
            observation, info = env.reset()
            traj = []
            obs_buf_ep = []
            act_buf_ep = []
            rews_buf_ep = []
            adv_buf_ep = []
            ret_buf_ep = []
            val_buf_ep = []
            logp_buf_ep = []
            num_steps = 0
            while True:
                action, val, logp = self.ac.step(torch.as_tensor(observation, dtype=torch.float32))
                
                obs_buf_ep.append(np.expand_dims(observation, axis=0))
                act_buf_ep.append(np.expand_dims(action, axis=0))
                logp_buf_ep.append(logp)
                val_buf_ep.append(val)

                observation, reward, terminated, truncated, info = env.step(action)

                rews_buf_ep.append(reward)
                num_steps += 1
                # traj_step = {"observation": observation, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}
                # traj.append(traj_step)
                if terminated or truncated or num_steps > self.args.max_ep_len:
                    # trajs.append(traj)
                    obs_buf_ep = np.concatenate(obs_buf_ep, axis=0)
                    act_buf_ep = np.concatenate(act_buf_ep, axis=0)
                    
                    rews = np.array(rews_buf_ep + [0])
                    vals = np.array(val_buf_ep + [0])
                    
                    rews_buf_ep = np.array(rews_buf_ep)
                    val_buf_ep = np.array(val_buf_ep)
                    logp_buf_ep = np.array(logp_buf_ep)
                    
                    # compute rewards
                    # compute advantage estimates on collected trajectories
                    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
                    adv_buf_ep = core.discount_cumsum(deltas, self.gamma * self.lam)
                    ret_buf_ep = core.discount_cumsum(rews, self.gamma)[:-1]

                    obs_buf.append(obs_buf_ep)
                    act_buf.append(act_buf_ep)
                    rews_buf.append(rews_buf_ep)
                    val_buf.append(val_buf_ep)
                    logp_buf.append(logp_buf_ep)
                    adv_buf.append(adv_buf_ep)
                    ret_buf.append(ret_buf_ep)

                    break
        buf = {}
        buf['obs'] = np.concatenate(obs_buf, axis=0)
        buf['act'] = np.concatenate(act_buf, axis=0)
        buf['rews'] = np.concatenate(rews_buf)
        buf['val'] = np.concatenate(val_buf)
        buf['logp'] = np.concatenate(logp_buf)
        buf['adv'] = np.concatenate(adv_buf)
        buf['ret'] = np.concatenate(ret_buf)
        buf['rews_total'] = np.array([np.sum(rews_buf_ep) for rews_buf_ep in rews_buf])
        buf['ep_len'] = np.array([len(rews_buf_ep) for rews_buf_ep in rews_buf])
        buf['rews_mean'] = np.array([np.mean(rews_buf_ep) for rews_buf_ep in rews_buf])
        del env
        print("deleted env, returning")
        return buf

    def combine_trajs(self, trajs):
        # for i in range(len(trajs[0])):
        #     assert np.allclose(trajs[0][i]["observation"], trajs[1][i]["observation"])
        # print([len(trajs[0][i]) for i in range(len(trajs[0]))])
        self.rollout_buffer = {}
        for k in trajs[0].keys():
            self.rollout_buffer[k] = np.concatenate([trajs[i][k] for i in range(len(trajs))])
        
        self.rollout_buffer['adv'] = (self.rollout_buffer['adv'] - np.mean(self.rollout_buffer['adv'])) / np.std(self.rollout_buffer['adv'])
        self.rollout_buffer = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in self.rollout_buffer.items()}

        # buf = self.rollout_buffer
        # newlogp = self.ac.pi(buf['obs'], buf['act'])[1]
        # assert torch.allclose(buf['logp'], newlogp), f"{buf['logp'].shape}, {newlogp.shape},\n logp={buf['logp']}, newlogp={newlogp}"

        # for k,v in self.rollout_buffer.items():
        #     print(f"{k}.shape={v.shape}")

    def collect_trajectories(self):
        # use multiprocessing to collect trajectories in parallel
        self.print_time("starting trajectory collection now")
        start_time = time.time()
        
        with Pool(self.num_envs) as p:
            trajs = p.map(self._collect_trajectories_sequential, [i for i in range(self.num_envs)])
        # trajs = [self._collect_trajectories_sequential(i) for i in range(self.num_envs)]

        self.combine_trajs(trajs)

        end_time = time.time()
        self.print_time(f"trajectory collection time = {end_time-start_time}s")
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(self):
        data = self.rollout_buffer
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info
    
    # Set up function for computing value loss
    def compute_loss_v(self):
        data = self.rollout_buffer
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret)**2).mean()

    def update_policy(self):
        # update weights of the policy network
        # Train policy with multiple steps of gradient descent
        self.print_time("starting update pi now")
        start_time = time.time()

        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi()
            self.pi_steps += 1
            kl = pi_info['kl']
            if kl > self.kl_multiplier * self.target_kl:
                print(f'Early stopping at step {i} due to reaching max kl={kl}, greater than {self.kl_multiplier * self.target_kl}.')
                break
            loss_pi.backward()
            self.pi_optimizer.step()
        end_time = time.time()
        self.print_time(f"update pi time = {end_time-start_time}s")
        # Log changes from update
        # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
    
    def update_value(self):
        self.print_time("starting update vf now")
        start_time = time.time()

        # update value function
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v()
            loss_v.backward()
            self.vf_optimizer.step()
            self.v_steps += 1
        end_time = time.time()
        self.print_time(f"update vf time = {end_time-start_time}s")

    def log_step(self):
        # collect trajectories
        self.print_time("starting logging now")
        start_time = time.time()

        loss_pi, _ = self.compute_loss_pi()
        loss_v = self.compute_loss_v()
        self.loss_pi, self.loss_v = loss_pi.item(), loss_v.item()
        gif_path = self.save_gif()
        wandb.log({"reward/avg reward vs number of episodes": torch.mean(self.rollout_buffer['rews_mean']).cpu().numpy(),
                "reward/total reward (averaged over multiple episodes) vs number of episodes": torch.mean(self.rollout_buffer['rews_total']).cpu().numpy(),
                "reward/episode length (averaged over multiple episodes) vs number of episodes": torch.mean(self.rollout_buffer['ep_len']).cpu().numpy(),
                "reward/episodes": self.episodes, "reward/video": wandb.Video(gif_path), "v/loss_v": self.loss_v, "v/step": self.v_steps,
                "pi/loss": self.loss_pi, "pi/step": self.pi_steps, **{f"pi/log_std[{i}]": val.item() for i,val in enumerate(self.ac.pi.log_std)}
                }
            )
        
        end_time = time.time()
        self.print_time(f"logging time = {end_time-start_time}s")

    def update_step(self):
        # update policy
        self.update_policy()
        # fit value function
        self.update_value()

    def train(self):
        # run train_step num_train times
        while self.episodes < self.total_episodes:
            self.collect_trajectories()

            if self.episodes % self.log_episodes == 0:
                self.log_step()

            self.update_step()

            self.episodes += self.episodes_per_epoch

    def save_gif(self):
        env = env_fn(self.bvh_path)
        o, info = env.reset()
        frames = [env.render()]
        dir_ = f'gifs-{self.exp_name}'
        
        if not os.path.isdir(dir_):
            os.mkdir(dir_)

        path = f'{dir_}/episodes_{self.episodes}.gif'
        num_steps = 0
        while True:
            num_steps += 1
            a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32))
            o, r, terminated, truncated, info = env.step(a)
            frames.append(env.render())
            if terminated or truncated or num_steps > self.args.max_ep_len:
                imageio.mimsave(path, frames)
                break
        
        # print("last joint positions")
        # model = env.agent_model
        # for model_link_id in model.get_link_ids():
        #     joint_type = model.get_link_joint_type(model_link_id)

        #     if joint_type == phy.JointType.Fixed:
        #         continue

        #     joint_name = model.get_link_name(model_link_id)
        #     offset = model.get_link_joint_pos_offset(model_link_id)
        #     print(f"name={joint_name}, link_id={model_link_id}, offset={offset}, pos/angle={model.joint_positions[offset]}")
        imageio.imwrite(f'{dir_}/episodes_{self.episodes}.png', frames[0])
        return path

    def print_time(self, *args, **kwargs):
        if self.show_time:
            print(*args, **kwargs)

class Dict2Class(object): 
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])

if __name__ == '__main__':
    args_dict = json.load(open("config/walker.json", "r"))
    args = Dict2Class(args_dict)

    wandb.init(project=args.project_name, name=args.exp_name, reinit=False, config=args_dict, mode=args.wandb_mode)

    wandb.define_metric("pi/step")
    wandb.define_metric("pi/*", step_metric="pi/step")
    wandb.define_metric("v/step")
    wandb.define_metric("v/*", step_metric="v/step")
    wandb.define_metric("reward/episodes")
    wandb.define_metric("reward/*", step_metric="reward/episodes")
    
    experiment_dir = pathlib.Path(args.experiment_dir)
    args.bvh_path = list(experiment_dir.glob("*.bvh"))[0].as_posix()
    
    with open(experiment_dir.joinpath("vr_data.json")) as f:
        vr_data = json.load(f)

    start, end = vr_data["bvh_trim_indices"]

    
    ppo = PPO(env_fn, args, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l))
    # ppo.save_gif()
    ppo.train()

# Command to run silently (no habitat-sim logging)
# export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet && python ppo_clip.py
