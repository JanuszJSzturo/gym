import random

import gym
from gym import spaces
import pygame
import numpy as np
from gym.utils.play import play
import copy
from enum import Enum, auto
import time

WALL_KICKS_JLTSZ = ([[[(-1, 0), (-1, 1), (0, -2), (-1, -2)], [(1, 0), (1, 1), (0, -2), (1, -2)]],
                     [[(1, 0), (1, -1), (0, 2), (1, 2)], [(1, 0), (1, -1), (0, 2), (1, 2)]],
                     [[(1, 0), (1, 1), (0, -2), (1, -2)], [(-1, 0), (-1, 1), (0, -2), (-1, -2)]],
                     [[(-1, 0), (-1, -1), (0, 2), (-1, 2)], [(-1, 0), (-1, -1), (0, 2), (-1, 2)]]])

WALL_KICKS_I = ([[[(-2, 0), (1, 0), (-2, -1), (1, 2)], [(-1, 0), (2, 0), (-1, 2), (2, -1)]],
                 [[(-1, 0), (2, 0), (-1, 2), (2, -1)], [(2, 0), (-1, 0), (2, 1), (-1, -2)]],
                 [[(2, 0), (-1, 0), (2, 1), (-1, -2)], [(1, 0), (-2, 0), (1, -2), (-2, 1)]],
                 [[(1, 0), (-2, 0), (1, -2), (-2, 1)], [(-2, 0), (1, 0), (-2, -1), (1, 2)]]])


class Action(Enum):
    # Change to another number for human player as 0 is active if no key is pressed
    LEFT = 99
    RIGHT = 1
    DOWN = 2
    ROT_LEFT = 3
    ROT_RIGHT = 4
    DROP = 5
    RESERVE = 6

# TODO: Comment the code
# TODO: test the game mechanics
# TODO: Define reward function
# TODO: Define observations
# TODO: implement score system
# TODO: configure automatic block falling if possible and speed up according lines cleared
class TetrisState:
    def __init__(self):
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
        self.reserved_tetr = None
        self.next_tetr = []
        self.can_reserve = True
        self.time = time.time()
        self.time_step = 1

        self.lines = 0
        self.score = 0
        self.pieces_placed = 0
        # testing editing from github

        self.actions_done = np.zeros(100)
        self.piece_epoch = 0

        for n in range(6):
            self.next_tetr.append(Tetromino.make())

    def update(self, action):
        over = False
        bottom_reached = False
        dt = time.time()-self.time

        if action == Action.LEFT.value:
            bottom_reached = self.current_tetr.move((-1, 0), self.played_tetr)
        elif action == Action.RIGHT.value:
            bottom_reached = self.current_tetr.move((1, 0), self.played_tetr)
        elif action == Action.DOWN.value:
            bottom_reached = self.current_tetr.move((0, 1), self.played_tetr)
        elif action == Action.ROT_LEFT.value:
            self.current_tetr.rotate(1, self.played_tetr)
        elif action == Action.ROT_RIGHT.value:
            self.current_tetr.rotate(-1, self.played_tetr)
        elif action == Action.DROP.value:
            self._drop()
            bottom_reached = True
        elif action == Action.RESERVE.value:
            self._reserve()

        # TODO: how should be the priority in gravity over actions?
        if dt >= self.time_step:
            bottom_reached = self.current_tetr.move((0, 1), self.played_tetr)
            self.time = time.time()

        if action != 0:
            if len(self.actions_done) <= self.piece_epoch:
                print(f'Piece dropped due to surpass maximum actions permited: {len(self.actions_done)}')
                self._drop()
                bottom_reached = True
            else:
                self.actions_done[self.piece_epoch] = action
                self.piece_epoch += 1

        if bottom_reached:
            # If current tetromino reached bottom we check line, spawn a new tetromino and check if it is game over
            self.pieces_placed += 1
            self.played_tetr.append(self.current_tetr)
            self._check_line()
            over = self._spawn_tetromino()
            self.can_reserve = True

        self._update_board()
        if over:
            print(self.score, self.lines, self.pieces_placed)
        return over, bottom_reached
    def _update_board(self):
        temp_board = np.zeros((21, 10))
        current_struct = np.where(self.current_tetr.struct)
        current_struct = current_struct[0] + self.current_tetr._y, current_struct[1] + self.current_tetr._x
        temp_board[current_struct] += 1
        for tetr in self.played_tetr:
            temp_struct = np.where(tetr.struct)
            temp_struct = temp_struct[0] + tetr._y, temp_struct[1] + tetr._x
            temp_board[temp_struct] += 1
        self.board = temp_board[:20]

    def _spawn_tetromino(self):
        self.actions_done = np.zeros(100)
        self.piece_epoch = 0
        tetr_name = self.next_tetr.pop(0).name
        self.next_tetr.append(Tetromino.make())

        self.current_tetr = Tetromino.make(tetr_name)
        terminated = Tetromino.collision(self.current_tetr, self.played_tetr)

        return terminated

    def _reserve(self):
        if self.reserved_tetr is None:
            self.reserved_tetr = Tetromino.make(self.current_tetr.name)
            self._spawn_tetromino()
        elif self.can_reserve:
            current_tetr_name = self.current_tetr.name
            self.current_tetr = Tetromino.make(self.reserved_tetr.name)
            self.reserved_tetr = Tetromino.make(current_tetr_name)
        self.can_reserve = False

    def _drop(self):
        bottom_reached = False
        while not bottom_reached:
            bottom_reached = self.current_tetr.move((0, 1), self.played_tetr)

    def _check_line(self):
        """
        The idea for checking and updating the grid when a line is complete is as follows:
        1. Indentify the rows where the line is complete (we do it if a row sums 10)
        2. For every row, we check every played tetromino if it is affected
            2.1. Get np.where(tetr.struct) and add the position of the tetr
            2.2. If any row of the tetr is equal to the row affected we change the 1's to 0's in the tetr struct
        3. We need to move down each tetromino accordingly
        """
        # List of tetrominoes that have to be removed when line is cleared
        tetr_to_remove = []

        # Check full rows
        board = self.get_board()
        affected_rows = np.where(board.sum(axis=1) == 10)[0]
        # Do nothing if there are no full lines
        if affected_rows.size != 0:
            self.lines += affected_rows.size
            self.score += affected_rows.size * 100
            for row in affected_rows:
                # Get all Tetrominoes that are affected by the row clear
                affected_tetr = TetrisState._tetr_in_row(self.played_tetr, row)
                for tetr in affected_tetr:
                    # Delete de row of the struct affected
                    tetr.struct = np.delete(tetr.struct, row-tetr._y, 0)
                    # If the struct is empty or all 0 we mark the piece to delete
                    if not tetr.struct.any():
                        tetr_to_remove.append(tetr)
                # Remove the marked pieces to remove from played tetrominoes and clear the list
                if tetr_to_remove:
                    for tetr in tetr_to_remove:
                        self.played_tetr.remove(tetr)
                    tetr_to_remove.clear()

                # Update the position for the affected tetrominoes by the line clear, i.e. all above the row.
                for tetr in self.played_tetr:
                    rows, _ = np.where(tetr.struct == 1)
                    real_y = tetr._y + min(rows)
                    if real_y <= row:
                        tetr.move((0, 1), self.played_tetr)

    @staticmethod
    def _tetr_in_row(tetrominoes, row):
        affected_tetr = []
        for tetr in tetrominoes:
            tetr_struct = np.where(tetr.struct)
            tetr_struct_shifted = tetr_struct[0] + tetr._y, tetr_struct[1] + tetr._x
            if np.any(tetr_struct_shifted[0] == row):
                affected_tetr.append(tetr)
        return affected_tetr

    def get_board(self):
        return self.board

    def get_reserved(self):
        return self.reserved_tetr

    def get_next_tetrominoes(self):
        return self.reserved_tetr

    def get_current_tetromino(self):
        return self.current_tetr

    def get_total_lines_cleared(self):
        return self.lines

    def get_score(self):
        return self.score

    def get_pieces_placed(self):
        return self.pieces_placed


class Tetromino:
    def __init__(self):
        self._rotation_state = 0
        self.struct = self.struct

    @staticmethod
    def make(name=None):
        """
        Returns a tetromino of type "name" or random tetromino if name is not specified
        name: name of the type of tetromino desired if None, random type is returned
        """
        if name == 'J':
            return JBlock()
        elif name == 'S':
            return SBlock()
        elif name == 'I':
            return IBlock()
        elif name == 'L':
            return LBlock()
        elif name == 'Z':
            return ZBlock()
        elif name == 'O':
            return OBlock()
        elif name == 'T':
            return TBlock()

        # If none type specified returns a random tetromino
        pool = (IBlock(), LBlock(), JBlock(), SBlock(), ZBlock(), OBlock(), TBlock())
        tetromino = random.choice(pool)
        return tetromino
    @staticmethod
    def remove_empty_rows(struct_array):
        new_array = np.copy(struct_array)
        rows, _ = np.where(new_array == 1)
        y_offset = min(rows)
        new_array = new_array[~np.all(new_array == 0, axis=1)]
        return new_array, y_offset

    def collision(self, others):
        """
        Return True if a tetromino collides with other in others, else return False
        """
        rows, cols = np.where(self.struct == 1)
        left = self._x + min(cols)
        right = self._x + max(cols)
        top = self._y + min(rows)
        bot = self._y + max(rows)

        if left < 0 or right > 9 or bot > 19:
            return True

        # Array that will be populated with all tetrominoes
        temp_board = np.zeros((21, 10))
        current_struct = np.where(self.struct)
        current_struct = current_struct[0] + self._y, current_struct[1] + self._x
        temp_board[current_struct] += 1
        for other in others:
            if other == self:
                continue
            temp_struct = np.where(other.struct)
            temp_struct = temp_struct[0] + other._y, temp_struct[1] + other._x
            temp_board[temp_struct] += 1
        # Check if any value is greater than 1, if it is then there is collision if there isn't any = no collision
            if np.any(temp_board > 1):
                return True
        return False

    def rotate(self, rot_direction, collision_group):
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

        if Tetromino.collision(self, collision_group):
            valid = False
        else:
            valid = True
        "At this point we have tried the basic rotation"
        if not valid:
            for move_x, move_y in wall_kicks[starting_rotation_state][rotation_direction]:
                self._x += move_x
                self._y -= move_y
                if not Tetromino.collision(self, collision_group):
                    valid = True
                    break
                else:
                    self._x -= move_x
                    self._y += move_y

        if not valid:
            self.struct = np.rot90(self.struct, k=-rot_direction)
            self._rotation_state = (self._rotation_state + rot_direction) % 4

    def move(self, direction, collision_group):
        bottom_reached = False
        # Resolve movement
        self._x += direction[0]
        self._y += direction[1]
        # If collision, undo movement
        if self.collision(collision_group):
            self._x -= direction[0]
            self._y -= direction[1]
            # If moving down, it means that the tetr reached bottom
            if direction[1]:
                bottom_reached = True

        return bottom_reached

    def get_state(self):
        return self._x, self._y, self._rotation_state

    def update_struct(self, row):
        new_struct = self.struct
        new_struct = np.delete(new_struct, row, 0)

        return new_struct


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
    _x = 3
    _y = -1


class LBlock(Tetromino):
    struct = np.array(
        ((0, 0, 1),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (237, 96, 0)
    name = 'L'
    rotations = 4
    _x = 3
    _y = -1


class JBlock(Tetromino):
    struct = np.array(
        ((1, 0, 0),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (41, 18, 245)
    name = 'J'
    rotations = 4
    _x = 3
    _y = -1


class OBlock(Tetromino):
    struct = np.array(
        ((1, 1),
         (1, 1))
    )
    color = (222, 165, 0)
    name = 'O'
    rotations = 0
    _x = 4
    _y = -1


class SBlock(Tetromino):
    struct = np.array(
        ((0, 1, 1),
         (1, 1, 0),
         (0, 0, 0))
    )
    color = (82, 226, 0)
    name = 'S'
    rotations = 4
    _x = 3
    _y = -1


class TBlock(Tetromino):
    struct = np.array(
        ((0, 1, 0),
         (1, 1, 1),
         (0, 0, 0))
    )
    color = (168, 23, 236)
    name = 'T'
    rotations = 4
    _x = 3
    _y = -1


class ZBlock(Tetromino):
    struct = np.array(
        ((1, 1, 0),
         (0, 1, 1),
         (0, 0, 0))
    )
    color = (249, 38, 52)
    name = 'Z'
    rotations = 4
    _x = 3
    _y = -1


class DotBlock(Tetromino):
    struct = np.array(
        ((0, 0, 0),
         (0, 1, 0),
         (0, 0, 0))
    )
    color = (87, 87, 87)
    name = 'Dot'
    rotations = 0
    _x = 3
    _y = -1


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
                "main_board": spaces.MultiBinary([20, 10]),
                "next_board": spaces.MultiDiscrete([7, 7, 7, 7, 7, 7]),
                "reserved_board": spaces.Discrete(7)
            }
        )

        # We have 7 actions, corresponding to "left", "right", "down", "rotate_left", "rotate_right", "drop", "reserve"
        self.action_space = spaces.Discrete(7)

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
        return {"main_board": self.internal_state.get_board(),
                "next_board": self.internal_state.get_next_tetrominoes(),
                "reserved_board": self.internal_state.get_reserved()}

    def _get_info(self):
        return {"total_lines_cleared": self.internal_state.get_total_lines_cleared(),
                "score": self.internal_state.get_score(),
                "bottom_reached": self.internal_state.get_bottom_reached()}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.internal_state.reset()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in

        # self.internal_state.update(action)

        # TODO define "terminated", "reward", "observation" and "info"
        terminated = self.internal_state.update(action)
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

        def render_tetr(tetromino, surface, cell_size, main_grid=True, pos_offset=(0, 0)):
            """
            Function to draw tetrominos on differents grid of gameplay.
            tetromino: tetromino to draw on surface
            surface: surface to draw tetromino onto
            cell_size: size in pixels of single square of a tetromino
            main_grid: bool to know if drawing tetromino on main grid or reserved/next grid
            pos_offset: offset in position when drawing on next grid.
            """
            off_set = 4
            board_position = (0, 0)
            if main_grid:
                x, y, _ = tetromino.get_state()
            else:
                x, y = 0, 0
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


        # Render current tetromino
        playing_tetr = self.internal_state.current_tetr
        render_tetr(playing_tetr, main_grid, self.cell_size)


        # Render played tetrominoes
        if self.internal_state.played_tetr:
            for tetr in self.internal_state.played_tetr:
                render_tetr(tetr, main_grid, self.cell_size)

        # Render reserved tetromino
        if self.internal_state.reserved_tetr:
            render_tetr(self.internal_state.reserved_tetr, reserve_grid, self.cell_size//2, False)

        # Render next tetrominoes
        for i, tetr in enumerate(self.internal_state.next_tetr):
            off_set = 3*i * self.cell_size//2
            render_tetr(tetr, next_grid, self.cell_size//2, False, (0, off_set))


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
    mapping = {(pygame.K_RIGHT,): 1,
               (pygame.K_DOWN,): 2,
               (pygame.K_LEFT,): 99,
               (pygame.K_a,): 3,
               (pygame.K_s,): 4,
               (pygame.K_UP,): 5,
               (pygame.K_r,): 6}
    play(env, keys_to_action=mapping)
