import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.ratio = self.size/(5*6)
        self.window_size = 508  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(5)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.vert_speed = 0
        self.horiz_speed = 0
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, 0]),
            4: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.pre_obs = [np.array([self.size-1, self.size, -self.size, 0])]

        self.stones = []
        self.ground = []
        self.pipes = []


        self.ground += [np.array([self.size-1, self.size, 0, self.size])]
        self.stones += [np.array([self.size-3, self.size-2.5, 5, 5.5]),
                       np.array([self.size-3, self.size-2.5, 7, 9.5]), np.array([self.size-5, self.size-4.5, 8, 8.5])]
        
        step = self.size
        self.ground += [np.array([self.size-1, self.size, 0+step, self.size+step])]
        self.pipes += [np.array([self.size-2, self.size-1, step+1, step+2])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, 0+step, self.size+step])]
        self.pipes += [np.array([self.size-2.5, self.size-1, step+1, step+2])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, 0+step, self.size+step-2])]
        self.pipes += [np.array([self.size-3, self.size-1, step+1, step+2])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, 0+step, 8.5+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 4+step, 5.5+step]),
                       np.array([self.size-5, self.size-4.5, 5.5+step, 9+step])]
        
        step += self.size
        self.ground += [np.array([self.size-1, self.size, 0.5+step, self.size+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 3.5+step, 4+step]), np.array([self.size-3, self.size-2.5, 9.5+step, 10+step]),
                       np.array([self.size-5, self.size-4.5, 2+step, 4+step]), np.array([self.size-3, self.size-2.5, 6.5+step, 7.5+step])]
        
        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, self.size+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 1+step, 1.5+step]),
                       np.array([self.size-5, self.size-4.5, 1+step, 1.5+step]), np.array([self.size-3, self.size-2.5, 2.5+step, 3+step]),
                       np.array([self.size-3, self.size-2.5, 5+step, 5.5+step]), np.array([self.size-5, self.size-4.5, 6.5+step, 8+step])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, self.size+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 0.5+step, 1.5+step]), np.array([self.size-3, self.size-1, 6+step, 6.5+step]),
                       np.array([self.size-5, self.size-4.5, step, 2+step]), np.array([self.size-1.5, self.size-1, 3+step, 3.5+step]),
                       np.array([self.size-2, self.size-1, 3.5+step, 4+step]), np.array([self.size-2.5, self.size-1, 4+step, 4.5+step]),
                       np.array([self.size-3, self.size-1, 4.5+step, 5+step]), np.array([self.size-1.5, self.size-1, 7.5+step, 8+step]),
                       np.array([self.size-2, self.size-1, 7+step, 7.5+step]), np.array([self.size-2.5, self.size-1, 6.5+step, 7+step]),]
        
        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, 2.5+step]), np.array([self.size-1, self.size, 4+step, self.size+step])]
        self.stones += [np.array([self.size-1, self.size, step, 2.5+step]), np.array([self.size-1.5, self.size-1, step, 0.5+step]),
                       np.array([self.size-2, self.size-1, 0.5+step, 1+step]), np.array([self.size-2.5, self.size-1, 1+step, 1.5+step]),
                       np.array([self.size-3, self.size-1, 1.5+step, 2+step]), np.array([self.size-3, self.size-1, 2+step, 2.5+step]),
                       np.array([self.size-1.5, self.size-1, 5.5+step, 6+step]),
                       np.array([self.size-2, self.size-1, 5+step, 5.5+step]), np.array([self.size-2.5, self.size-1, 4.5+step, 5+step]),
                       np.array([self.size-3, self.size-1, 4+step, 4.5+step])]
        self.pipes += [np.array([self.size-2, self.size-1, step+8, step+9])]

        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, self.size+step])]
        self.pipes += [np.array([self.size-2, self.size-1, 6+step, 7+step])]
        self.stones += [np.array([self.size-3, self.size-2.5, 0.5+step, 2.5+step]), np.array([self.size-1.5, self.size-1, 7+step, 7.5+step]),
                       np.array([self.size-2, self.size-1, 7.5+step, 8+step]),np.array([self.size-2.5, self.size-1, 8+step, 8.5+step]),
                       np.array([self.size-3, self.size-1, 8.5+step, 9+step]),np.array([self.size-3.5, self.size-1, 9+step, 9.5+step]),
                       np.array([self.size-4, self.size-1, 9.5+step, 10+step])]
        
        step += self.size
        self.ground += [np.array([self.size-1, self.size, step, self.size+step])]
        self.stones += [np.array([self.size-4.5, self.size-1, step, 0.5+step]),
                       np.array([self.size-5, self.size-1, 0.5+step, 1+step]),np.array([self.size-5.5, self.size-1, 1+step, 1.5+step]),
                       np.array([self.size-1.5, self.size-1, 9.5+step, 10+step])]
        
        self.flag = [np.array([3, self.size-1.5, 9+step, 10+step])]


        self.obstacles = self.ground + self.stones + self.pipes

        self.monsters = [Monster(self.size-1.6, 7, self.obstacles), Monster(self.size-1.6, self.size+7, self.obstacles), Monster(self.size-1.6, 2*self.size+7, self.obstacles), Monster(self.size-1.6, 2*self.size+7.5, self.obstacles),
                         Monster(self.size-5.6, 4*self.size+5.5, self.obstacles), Monster(self.size-5.6, 4*self.size+6.5, self.obstacles),
                         Monster(self.size-1.6, 7.5+5*self.size, self.obstacles), Monster(self.size-1.6, 7+5*self.size, self.obstacles),
                         Monster(self.size-1.6, 6*self.size, self.obstacles), Monster(self.size-1.6, 5+6*self.size, self.obstacles),
                         Monster(self.size-1.6, 5.5+6*self.size, self.obstacles),
                         Monster(self.size-1.6, 7*self.size, self.obstacles), Monster(self.size-1.6, 0.5+7*self.size, self.obstacles),
                         Monster(self.size-1.6, 2+7*self.size, self.obstacles), Monster(self.size-1.6, 2.5+7*self.size, self.obstacles),
                         Monster(self.size-1.6, 4.5+9*self.size, self.obstacles), Monster(self.size-1.6, 5+9*self.size, self.obstacles)]

    
    def _get_info(self):
        return {
            "distance": self.max_dist
        }
    
    def update_position(self):
        self.top = self._agent_location[1] - 0.5
        self.bottom = self._agent_location[1]
        self.right = self._agent_location[0] + 0.5
        self.left = self._agent_location[0]

    def top_contact(self):
        for obstacle in self.obstacles:
            if self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]:
                if (self.right >= obstacle[2] and self.right < obstacle[3]):
                    self.vert_speed = obstacle[1] - self.top
                elif (self.left > obstacle[2] and self.left <= obstacle[3]):
                    self.vert_speed = obstacle[1] - self.top

    def bot_contact(self):
        for obstacle in self.obstacles:
            if self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]:
                if (self.right > obstacle[2] and self.right <= obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.left >= obstacle[2] and self.left < obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.right < obstacle[2] and self.right + self.horiz_speed >= obstacle[2]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.left > obstacle[3] and self.left + self.horiz_speed <= obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                
    def can_jump(self):
        for obstacle in self.obstacles:
            if self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]:
                if (self.right > obstacle[2] and self.right <= obstacle[3]):
                    return True
                elif (self.left >= obstacle[2] and self.left < obstacle[3]):
                    return True
        return False

    def right_contact(self):
        for obstacle in self.obstacles:
            if self.right <= obstacle[2] and self.right + self.horiz_speed > obstacle[2]:
                if (self.top < obstacle[1] and self.top >= obstacle[0]):
                    self.horiz_speed = obstacle[2] - self.right
                elif (self.bottom > obstacle[0] and self.bottom <= obstacle[1]):
                    self.horiz_speed = obstacle[2] - self.right
                elif (self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.right
                elif (self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]):
                    self.horiz_speed = obstacle[3] - self.right

    def left_contact(self):
        for obstacle in self.obstacles:
            if self.left >= obstacle[3] and self.left + self.horiz_speed < obstacle[3]:
                if (self.top < obstacle[1] and self.top >= obstacle[0]):
                    self.horiz_speed = obstacle[3] - self.left
                elif (self.bottom > obstacle[0] and self.bottom <= obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                elif (self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                elif (self.bottom <= obstacle[1] and self.bottom + self.vert_speed > obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.array([0, self.size-2])
        self.max_dist = self._agent_location[0]
        self.update_position()
        self.terminated = False
        for monster in self.monsters:
            monster.reset()

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
        observation = self._render_frame()
        info = self._get_info()

        return observation, info


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        jump_state = self.can_jump()
        if action in [3, 5, 6] and jump_state:
            self.vert_speed = -2.6 * self.ratio
        elif action in [0, 5] and (self.horiz_speed >= 0 or jump_state):
            self.horiz_speed = min(max(self.horiz_speed, 0) + 0.4*self.ratio, 1*self.ratio)
        elif action in [2, 6] and (self.horiz_speed <= 0 or jump_state):
            self.horiz_speed = max(min(self.horiz_speed, 0) - 0.4*self.ratio, -1*self.ratio)
        elif action == 4:
            self.horiz_speed = 0

        self.update_position()
        self.top_contact()
        self.bot_contact()
        self.right_contact()
        self.left_contact()

        for monster in self.monsters:
            monster.step(self)
            self.terminated = self.terminated or monster.player_left_contact(self) or monster.player_right_contact(self)
        
        direction = np.array([1, 0])*self.horiz_speed + np.array([0, 1])*self.vert_speed

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, 200*self.size
        )
        self.vert_speed += 10*0.06*self.ratio

        if self._agent_location[0] // 5 > self.max_dist // 5:
            reward = 0.1
        else:
            reward =  0  # Binary sparse rewards
        observation = self._render_frame()
        info = self._get_info()

        self.max_dist = max(self.max_dist, self._agent_location[0])

        if self._agent_location[1] >= self.size:
            self.terminated = True

        if self._agent_location[0] >= 10*self.size + 9.5:
            info["flag"] = True
        else:
            info["flag"] = False
        
        if info["flag"]:
            reward = 0.2
            self.reset()

        if self.terminated:
            self.reset()

        return observation, reward, self.terminated, info


    def render(self):
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
        # if self.render_mode == "rgb_array":
        #     return self._render_frame()

    def _render_frame(self, have_state=False):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        bg = pygame.image.load("./source/background.png")
        bg = pygame.transform.scale(bg, (self.window_size, self.window_size))
        canvas.blit(bg, (0, 0))
        pix_square_size = (
            self.window_size / 8 #self.size
        )  # The size of a single grid square in pixels

        for obstacle in self.flag:
            flag = pygame.image.load("./source/flag.png")
            flag = pygame.transform.scale(flag, (pix_square_size*(obstacle[3] - obstacle[2]), pix_square_size*(obstacle[1] - obstacle[0])))
            canvas.blit(flag, ((obstacle[2]-1 - self._agent_location[0] + self.size/2)*pix_square_size, (obstacle[0]-2)*pix_square_size))

        # Now we draw the agent
        agent = pygame.image.load("./source/mario.png")
        agent = pygame.transform.scale(agent, (pix_square_size / 2, pix_square_size / 2))
        canvas.blit(agent, (pix_square_size*(self.size/2-1), pix_square_size*(self._agent_location[1]-2.5)))

        ground = pygame.image.load("./source/ground.png")
        ground = pygame.transform.scale(ground, (pix_square_size / 2, pix_square_size / 2))

        for obstacle in self.pre_obs:
            for h in np.arange(obstacle[0], obstacle[1]-0.1, 0.5):
                for l in np.arange(obstacle[2], obstacle[3]-0.1, 0.5):
                    canvas.blit(ground, ((l - 1 - self._agent_location[0] + self.size/2)*pix_square_size, (h-2)*pix_square_size))


        for obstacle in self.ground:
            for h in np.arange(obstacle[0], obstacle[1]-0.1, 0.5):
                for l in np.arange(obstacle[2], obstacle[3]-0.1, 0.5):
                    canvas.blit(ground, ((l -1 - self._agent_location[0] + self.size/2)*pix_square_size, (h-2)*pix_square_size))

        stone = pygame.image.load("./source/stone.png")
        stone = pygame.transform.scale(stone, (pix_square_size / 2, pix_square_size / 2))
        for obstacle in self.stones:
            for h in np.arange(obstacle[0], obstacle[1]-0.1, 0.5):
                for l in np.arange(obstacle[2], obstacle[3]-0.1, 0.5):
                    canvas.blit(stone, ((l -1 - self._agent_location[0] + self.size/2)*pix_square_size, (h-2)*pix_square_size))

        for obstacle in self.pipes:
            pipe = pygame.image.load("./source/pipes.png")
            pipe = pygame.transform.scale(pipe, (pix_square_size*(obstacle[3] - obstacle[2]), pix_square_size*(obstacle[1] - obstacle[0])))
            canvas.blit(pipe, ((obstacle[2]-1 - self._agent_location[0] + self.size/2)*pix_square_size, (obstacle[0]-2)*pix_square_size))
                



        champi = pygame.image.load("./source/champi.png")
        champi = pygame.transform.scale(champi, (pix_square_size / 2, pix_square_size / 2))
        for monster in self.monsters:
            canvas.blit(champi, ((monster.left - 1 - self._agent_location[0] + self.size/2)*pix_square_size, (monster.top-2)*pix_square_size))
        
            

        self.canvas = canvas

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )



    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



class Monster():
    def __init__(self, top_pos, left_pos, obstacles):
        self.monster_pos = np.array([left_pos, top_pos])
        self.horiz_speed = -0.1
        self.last_speed = -0.1
        self.vert_speed = 0
        self.obstacles = obstacles
        self.contact = False
        self.init_pos = np.array([left_pos, top_pos])
        self.update_position()
        self.is_dead = False

    def update_position(self):
        self.top = self.monster_pos[1]
        self.bottom = self.monster_pos[1] + 0.5
        self.right = self.monster_pos[0] + 0.5
        self.left = self.monster_pos[0]

    def bot_contact(self):
        for obstacle in self.obstacles:
            if self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]:
                if (self.right > obstacle[2] and self.right <= obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.left >= obstacle[2] and self.left < obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.right < obstacle[2] and self.right + self.horiz_speed >= obstacle[2]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)
                elif (self.left > obstacle[3] and self.left + self.horiz_speed <= obstacle[3]):
                    self.vert_speed = min(obstacle[0] - self.bottom, self.vert_speed)

    def right_contact(self):
        for obstacle in self.obstacles:
            if self.right <= obstacle[2] and self.right + self.horiz_speed > obstacle[2]:
                if (self.top < obstacle[1] and self.top >= obstacle[0]):
                    self.horiz_speed = obstacle[2] - self.right
                    self.contact = True
                elif (self.bottom > obstacle[0] and self.bottom <= obstacle[1]):
                    self.horiz_speed = obstacle[2] - self.right
                    self.contact = True
                elif (self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.right
                    self.contact = True
                elif (self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]):
                    self.horiz_speed = obstacle[3] - self.right
                    self.contact = True

    def left_contact(self):
        for obstacle in self.obstacles:
            if self.left >= obstacle[3] and self.left + self.horiz_speed < obstacle[3]:
                if (self.top < obstacle[1] and self.top >= obstacle[0]):
                    self.horiz_speed = obstacle[3] - self.left
                    self.contact = True
                elif (self.bottom > obstacle[0] and self.bottom <= obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                    self.contact = True
                elif (self.top >= obstacle[1] and self.top + self.vert_speed < obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                    self.contact = True
                elif (self.bottom <= obstacle[1] and self.bottom + self.vert_speed > obstacle[1]):
                    self.horiz_speed = obstacle[3] - self.left
                    self.contact = True
    
    def reset(self):
        self.monster_pos = np.copy(self.init_pos)
        self.is_dead = False
        self.horiz_speed = -0.1
        self.last_speed = -0.1
        self.contact = False
        self.update_position()

    def can_jump(self):
        for obstacle in self.obstacles:
            if self.bottom <= obstacle[0] and self.bottom + self.vert_speed > obstacle[0]:
                if (self.right > obstacle[2] and self.right <= obstacle[3]):
                    return True
                elif (self.left >= obstacle[2] and self.left < obstacle[3]):
                    return True
        return False

    def step(self, agent):
        if not self.is_dead and np.abs(agent.right - self.left) < agent.size:
            self.bot_contact()
            self.right_contact()
            self.left_contact()
            direction = np.array([1, 0])*self.horiz_speed + np.array([0, 1])*self.vert_speed
            self.monster_pos += direction
            self.update_position()
            self.vert_speed += 10*0.06/3

            self.is_dead = self.player_top_contact(agent)
            if self.monster_pos[1]+0.5 >= 10:
                self.is_dead = True

            if self.contact:
                self.horiz_speed = -1*self.last_speed
                self.last_speed *= -1
                self.contact = False
            
            if self.is_dead:
                self.monster_pos = np.array([-1, -1])
                self.update_position()

    def player_top_contact(self, player):
        if player.bottom <= self.top and player.bottom + player.vert_speed > self.top:
            if (player.right > self.left and player.right <= self.right):
                return True
            elif (player.left >= self.left and player.left < self.right):
                return True
            elif (player.right < self.left and player.right + player.horiz_speed >= self.left):
                return True
            elif (player.left > self.right and player.left + player.horiz_speed <= self.right):
                return True
            return False
        return False
    
    def player_right_contact(self, player):
        if player.right <= self.left and player.right + player.horiz_speed > self.left+self.horiz_speed:
            if (player.top <= self.bottom and player.top >= self.top):
                return True
            elif (player.bottom >= self.top and player.bottom <= self.bottom):
                return True
            elif (player.top >= self.bottom and player.top + player.vert_speed <= self.bottom):
                return True
            elif (player.bottom <= self.top and player.bottom + player.vert_speed >= self.top):
                return True
            return False
        return False

    def player_left_contact(self, player):
        if player.left >= self.right and player.left + player.horiz_speed < self.right+self.horiz_speed:
            if (player.top <= self.bottom and player.top >= self.top):
                return True
            elif (player.bottom >= self.top and player.bottom <= self.bottom):
                return True
            elif (player.top >= self.bottom and player.top + player.vert_speed <= self.bottom):
                return True
            elif (player.bottom <= self.top and player.bottom + player.vert_speed >= self.top):
                return True
            return False
        return False