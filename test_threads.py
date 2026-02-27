import sys
from config import Config
from train import training_loop

config = Config()
config.num_simulations = 40
config.num_self_play_games = 8 
config.self_play_threads = 8
config.inference_batch_size = 4
config.num_epochs = 1
config.num_res_blocks = 2 

if __name__ == "__main__":
    training_loop(config, num_iterations=1)
