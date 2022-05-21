from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from multiprocessing import freeze_support
import os
from sc2env import Sc2Env

import time
from datetime import datetime
from wandb.integration.sb3 import WandbCallback
import wandb

def make_env(map_shape, env_id, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = Sc2Env(map_shape, env_id=env_id)
        # Important: use a different seed for each environment
        env.seed(seed)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
	freeze_support()
	#sentdex's code
	# run = wandb.init(
	#     project=f'SC2RLv6',
	#     entity="sentdex",
	#     config=conf_dict,
	#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
	#     save_code=True,  # optional
	# )


	# run = wandb.init(
	#     project=f'SC2RLv1',
	#     entity="green-cabbage",
	#     config=conf_dict,
	#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
	#     save_code=True,  # optional
	# )

	map_name ="KingsCoveLE"
	map_shape = (176, 176, 3) # map shape for KingsCoveLE

	num_cpu = 1 # Number of processes to use
	if num_cpu==1: # single thread/process
		env = Sc2Env(map_shape)
	else:# multi thread/process
		# Create the vectorized environment
		env = SubprocVecEnv([make_env(map_shape, i) for i in range(num_cpu)])

	
	start_from_scratch = True
	if start_from_scratch:
		print("starting from scratch")
		iters = 0
		model_name = f"{int(time.time())}_cpu{num_cpu}"

		models_dir = f"models/{model_name}/"

	else: # continue from where we left off
		print("continuing from where we left off")
		iters=86
		model_name = "1653074223_cpu1"
		models_dir = f"models/{model_name}/"
		
		

	logdir = f"logs/{model_name}/"
	conf_dict = {"Model": "v1",
				"Machine": "Main",
				"policy":"MlpPolicy",
				"model_save_name": model_name}
	if not os.path.exists(models_dir):
		os.makedirs(models_dir)

	if not os.path.exists(logdir):
		os.makedirs(logdir)

	model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device="cuda:1")
	if not start_from_scratch:
		print("loading model")
		model.load(f"{models_dir}/1720000_Iter{iters}")

	TIMESTEPS = 20000
	with open(f"time.txt","w") as f: #prev on w mode
			f.write(f"cpu_count:{num_cpu}\n")

	
	while True:
		print("On iteration: ", iters)
		#save date
		with open(f"time.txt","a") as f:
			now = datetime.now()
			f.write(f"ITERATION_{iters}_DATE" + \
				now.strftime("%H:%M:%S") + "\n")
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
		model.save(f"{models_dir}/{TIMESTEPS*iters}_Iter{iters}")
