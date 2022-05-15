from stable_baselines3 import PPO
import os
from sc2env import Sc2Env
import time
from wandb.integration.sb3 import WandbCallback
import wandb


model_name = f"{int(time.time())}"

models_dir = f"models/{model_name}/"
logdir = f"logs/{model_name}/"


conf_dict = {"Model": "v1",
             "Machine": "Main",
             "policy":"MlpPolicy",
             "model_save_name": model_name}

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



if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

map_name ="KingsCoveLE"
map_shape = (176, 176, 3) # map shape for KingsCoveLE
env = Sc2Env(map_shape)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir, device="cuda:1")
# # start from scratch
# iters = 0
# continue from where we left off
iters=1
model_name = 1652650075
models_dir = f"models/{model_name}/"
model.load(f"{models_dir}/10000_Iter1")

TIMESTEPS = 10000

while True:
	print("On iteration: ", iters)
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}_Iter{iters}")
