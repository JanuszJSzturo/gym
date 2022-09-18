import random

import gym
from gym import spaces
import pygame
import numpy as np
from gym.utils.play import play
import copy


WALL_KICKS_JLTSZ = ([[[(-1, 0), (-1, 1), (0, -2), (-1, -2)], [(1, 0), (1, 1), (0, -2), (1, -2)]],
                     [[(1, 0), (1, -1), (0, 2), (1, 2)], [(1, 0), (1, -1), (0, 2), (1, 2)]],
                     [[(1, 0), (1, 1), (0, -2), (1, -2)], [(-1, 0), (-1, 1), (0, -2), (-1, -2)]],
                     [[(-1, 0), (-1, -1), (0, 2), (-1, 2)], [(-1, 0), (-1, -1), (0, 2), (-1, 2)]]])

WALL_KICKS_I = ([[[(-2, 0), (1, 0), (-2, -1), (1, 2)], [(-1, 0), (2, 0), (-1, 2), (2, -1)]],
                 [[(-1, 0), (2, 0), (-1, 2), (2, -1)], [(2, 0), (-1, 0), (2, 1), (-1, -2)]],
                 [[(2, 0), (-1, 0), (2, 1), (-1, -2)], [(1, 0), (-2, 0), (1, -2), (-2, 1)]],
                 [[(1, 0), (-2, 0), (1, -2), (-2, 1)], [(-2, 0), (1, 0), (-2, -1), (1, 2)]]])


class TetrisState:
    def __init__(self):
        # self.board = np.zeros((20, 10))
        # self.current_tetr = None
        # self.played_tetr = []
        # self.reserved_tetr = []
        # self.next_tetr = []

        self.reset()

    def reset(self):
        """
        Initialize a tetris game state
        clean the board
        clean reserved tetromino
        clean played tetrominoes
        initializes current tetromino with a random tetromino
        populate future tetrominoes with 6 random tetromino
        """
        self.board = np.zeros((20, 10))
        self.current_tetr = Tetromino.make()
        self.played_tetr = []
        self.reserved_tetr = []
        self.next_tetr = []

        for n in range(6):
            self.next_tetr.append(Tetromino.make())


    def update(self, action):
        return




    def spawn_tetromino(self):
        tetromino = self.next_tetr.pop()
        return tetromino


class Tetromino:
    def __init__(self):
        self._x = 0
        self._y = 0
        self._rotation_state = 0

    @staticmethod
    def make():
        pool = (IBlock(), LBlock(), JBlock(), SBlock(), ZBlock(), OBlock(), TBlock())
        tetromino = random.choice(pool)
        return tetromino

    def collision(self, others):
        # TODO: Where should be collision? in Tetromino class or TetrisState??
        """
        Return True if a tetromino collides with other in others, else return False
        """

        # Array that will be populated with all tetrominoes
        temp_board = np.zeros((20, 10))
        current_struct = np.where(self.struct)
        current_struct = current_struct[0] + self._y, current_struct[1] + self._x
        temp_board[current_struct] += 1
        for other in others:
            temp_struct = np.where(other.struct)
            temp_struct = temp_struct[0] + other._y, temp_struct[1] + other._x
            temp_board[temp_struct] += 1
        # Check if any value is greater than 1, if it is then there is collision if there isn't any = no collision
            if np.any(temp_board > 1):
                return True
        return False

    def rotate(self, rot_direction):
        # TODO: Configure rotate with the new class of Tetromino
        """
        rot_direction = 1 or -1 to rotate left or right respectively
        self.struct = np.rot90(self.struct, k=rotation)
        self.rotation_state = (self.rotation_state - rotation) % 4
        self.update()
        """

        # Select the wall kick array according to the piece
        if self.name == 'I':
            wall_kicks = WALL_KICKS_I
        else:
            wall_kicks = WALL_KICKS_JLTSZ

        starting_rotation_state = copy.copy(self._rotation_state)
        rotation_direction = (rot_direction + 1) // 2  # 0-> clockwise, 1-> counter-clockwise"
        self.struct = np.rot90(self.struct, k=rot_direction)
        self._rotation_state = (self._rotation_state - rot_direction) % 4

        # self.update()

        # if MyTetromino.collides(self, group) or self.out_of_board():
        #     valid = False
        # else:
        #     valid = True
        # "At this point we have tried the basic rotation"
        # if not valid:
        #     for move_x, move_y in wall_kicks[starting_rotation_state][rotation_direction]:
        #         self.x += move_x
        #         self.y -= move_y
        #         if not MyTetromino.collides(self, group) and not self.out_of_board():
        #             valid = True
        #             print("Wall kick performed: ", move_x, move_y)
        #             break
        #         else:
        #             self.x -= move_x
        #             self.y += move_y
        #
        # if not valid:
        #     self.struct = np.rot90(self.struct, k=-rotation)
        #     self.rotation_state = (self.rotation_state + rotation) % 4
        #     self.update()

    def move(self, direction):
        self._x += direction[0]
        self._y += direction[1]

    def get_state(self):
        return self._x, self._y, self._rotation_state


class IBlock(Tetromino):
    struct = np.array(
        ((0, 0, 0, 0),
         (1, 1, 1, 1),
         (0, 0, 0, 0),
         (0, 0, 0, 0))
    )
    color = (0, 168, 221)
    name = 'I'
    rotations = 4


class LBlock(Tetromino):
    struct = np.array(
        ((0, 0, 1),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (237, 96, 0)
    name = 'L'
    rotations = 4


class JBlock(Tetromino):
    struct = np.array(
        ((1, 0, 0),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (41, 18, 245)
    name = 'J'
    rotations = 4


class OBlock(Tetromino):
    struct = np.array(
        ((1, 1),
         (1, 1))
    )
    color = (222, 165, 0)
    name = 'O'
    rotations = 0


class SBlock(Tetromino):
    struct = np.array(
        ((0, 1, 1),
         (1, 1, 0),
         (0, 0, 0))
    )
    color = (82, 226, 0)
    name = 'S'
    rotations = 4


class TBlock(Tetromino):
    struct = np.array(
        ((0, 1, 0),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (168, 23, 236)
    name = 'T'
    rotations = 4


class ZBlock(Tetromino):
    struct = np.array(
        ((1, 1, 0),
         (0, 1, 1),
         (0, 0, 0))
    )
    color = (249, 38, 52)
    name = 'Z'
    rotations = 4


class DotBlock(Tetromino):
    struct = np.array(
        ((0, 0, 0),
         (0, 1, 0),
         (0, 0, 0))
    )
    color = (87, 87, 87)
    name = 'Dot'
    rotations = 4



class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 24}

    def __init__(self, render_mode=None, size=32):
        self.internal_state = TetrisState()
        self.cell_size = size  # The size of a square cell
        self.height = 23  # Height in cells of PyGame window
        self.width = 18  # Width in cells of PyGame window
        self.window_size = np.array((self.width * self.cell_size, self.height * self.cell_size))  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.cell_size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.cell_size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1])
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

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        x = int(self.np_random.integers(0, self.width, size=1, dtype=int))
        y = int(self.np_random.integers(0, self.height, size=1, dtype=int))
        self._agent_location = np.array((y, x))

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            x = int(self.np_random.integers(0, self.width, size=1, dtype=int))
            y = int(self.np_random.integers(0, self.height, size=1, dtype=int))
            self._target_location = np.array((y, x))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.cell_size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((0, 0, 0))

        main_grid = pygame.Surface((self.cell_size*10 + 1, self.cell_size*20 + 1))
        main_grid.fill((0, 0, 0))
        reserve_grid = pygame.Surface((self.cell_size*2 + 1, self.cell_size*1 + 1))
        reserve_grid.fill((0, 0, 0))
        next_grid = pygame.Surface((self.cell_size*2 + 1, self.cell_size*9 + 1))
        next_grid.fill((0, 0, 0))

        def draw_grid_lines(surface, spacing):
            """
            surface: destination surface where draw the lines
            spacing: pixels between lines (square grid)
            """
            height = surface.get_height() // spacing
            width = surface.get_width() // spacing
            for y in range(height + 1):
                pygame.draw.line(
                    surface=surface,
                    color=(255, 255, 255),
                    start_pos=(0, spacing * y),
                    end_pos=(width * spacing, spacing * y),
                    width=1,
                )
            for x in range(width + 1):
                pygame.draw.line(
                    surface=surface,
                    color=(255, 255, 255),
                    start_pos=(spacing * x, 0),
                    end_pos=(spacing * x, height * spacing),
                    width=1,
                )

        def render_tetr(tetromino, surface, cell_size, pos_offset=(0, 0)):
            off_set = 4
            board_position = (0, 0)
            x, y, _ = tetromino.get_state()
            rows, cols = np.where(tetromino.struct == 1)
            for row, col in zip(rows, cols):
                pygame.draw.rect(
                    surface=surface,
                    color=tetromino.color,
                    rect=pygame.Rect((col + x) * cell_size + off_set/2 + pos_offset[0],
                                     (row + y) * cell_size + off_set/2 + pos_offset[1],
                                     cell_size - off_set + 1,
                                     cell_size - off_set + 1)
                )

        # START OF TESTING #
        # TODO: delete after finishing the testing
        a = IBlock()
        a.move((0, 0))
        a.rotate(0)
        b = self.internal_state.next_tetr[0]
        render_tetr(a, main_grid, self.cell_size)

        for i, tetr in enumerate(self.internal_state.next_tetr):
            off_set = 3*i * self.cell_size//2
            render_tetr(tetr, next_grid, self.cell_size//2, (0, off_set))

        render_tetr(b, reserve_grid, self.cell_size // 2)
        # END OF TESTING #

        # Render next tetrominoes
        for i, tetr in enumerate(self.internal_state.next_tetr):
            off_set = 3*i * self.cell_size//2
            render_tetr(tetr, next_grid, self.cell_size//2, (0, off_set))

        # Finally, add some gridlines
        draw_grid_lines(main_grid, self.cell_size)
        draw_grid_lines(reserve_grid, self.cell_size//2)
        draw_grid_lines(next_grid, self.cell_size//2)

        # Blits the different surfaces onto canvas
        canvas.blit(source=main_grid, dest=(4*self.cell_size, 2*self.cell_size))
        canvas.blit(source=reserve_grid, dest=(1*self.cell_size, 2*self.cell_size))
        canvas.blit(source=next_grid, dest=(15*self.cell_size, 2*self.cell_size))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = TetrisEnv(render_mode="rgb_array")
    mapping = {(pygame.K_RIGHT,): 1, (pygame.K_DOWN,): 2, (pygame.K_LEFT,): 3, (pygame.K_UP,): 4}
    play(env, keys_to_action=mapping)
