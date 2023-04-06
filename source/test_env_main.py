from test_env import *
import gymnasium as gym
import time
import numpy as np
import keyboard

env = GridWorldEnv(render_mode=None, size=10)#gym.make("gym_examples/GridWorld-v0")
env.reset()

while True:
    # env.render()
    time.sleep(0.001)
    # if keyboard.is_pressed("z") and keyboard.is_pressed("d"):
    #     action = 5
    # elif keyboard.is_pressed("z") and keyboard.is_pressed("q"):
    #     action = 6
    if keyboard.is_pressed("z"):
        action = 3
    elif keyboard.is_pressed("s"):
        action = 1
    elif keyboard.is_pressed("d"):
        action = 0
    elif keyboard.is_pressed("q"):
        action = 2
    else:
        action = 4
    
    
    # action = np.random.randint(0, env.action_space.n)
    env.step(action)


