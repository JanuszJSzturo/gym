import random
import numpy as np
import pygame
from enum import Enum
import time
import copy


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
    LEFT = 0
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


        self.actions_done = np.zeros(100)
        self.current_piece_num_actions = 0
        self.last_piece_num_actions = 0

        for n in range(6):
            self.next_tetr.append(Tetromino.make())

    def update(self, action):
        over = False
        bottom_reached = False
        lines_cleared = 0
        dt = time.time()-self.time
        try:
            action = Action(action)
        except ValueError:
            action = None

        if action == Action.LEFT:
            bottom_reached = self.current_tetr.move((-1, 0), self.played_tetr)
        elif action == Action.RIGHT:
            bottom_reached = self.current_tetr.move((1, 0), self.played_tetr)
        elif action == Action.DOWN:
            bottom_reached = self.current_tetr.move((0, 1), self.played_tetr)
        elif action == Action.ROT_LEFT:
            self.current_tetr.rotate(1, self.played_tetr)
        elif action == Action.ROT_RIGHT:
            self.current_tetr.rotate(-1, self.played_tetr)
        elif action == Action.DROP:
            self._drop()
            bottom_reached = True
        elif action == Action.RESERVE:
            self._reserve()

        # TODO: how should be the priority in gravity over actions?
        if dt >= self.time_step:
            bottom_reached = self.current_tetr.move((0, 1), self.played_tetr)
            self.time = time.time()

        # if action != 0:
        if len(self.actions_done) <= self.current_piece_num_actions:
            print(f'Piece dropped due to surpass maximum actions permited: {len(self.actions_done)}')
            self._drop()
            bottom_reached = True
        elif action is not None:
            self.actions_done[self.current_piece_num_actions] = action.value
            self.current_piece_num_actions += 1

        if bottom_reached:
            # If current tetromino reached bottom we check line, spawn a new tetromino and check if it is game over
            self.pieces_placed += 1
            self.played_tetr.append(self.current_tetr)
            lines_cleared = self._check_line()
            over = self._spawn_tetromino()
            self.can_reserve = True

        self._update_board()

        if self.lines >= 100:
            print("Reached 100 lines cleared")
            over = True
        if over:
            print(self.score, self.lines, self.pieces_placed)
        return over, bottom_reached, lines_cleared
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
        self.last_piece_num_actions = self.current_piece_num_actions
        self.current_piece_num_actions = 0
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
        return affected_rows.size

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
        if self.reserved_tetr is None:
            return 0
        return self.reserved_tetr.name.value

    def get_next_tetrominoes(self):
        next_tetr = np.zeros_like(self.next_tetr)
        for i, tetr in enumerate(self.next_tetr):
            next_tetr[i] = tetr.name.value
        return next_tetr

    def get_current_tetromino(self):
        if self.current_tetr is None:
            return 0
        return self.current_tetr.name.value

    def get_total_lines_cleared(self):
        return self.lines

    def get_score(self):
        return self.score

    def get_pieces_placed(self):
        return self.pieces_placed

    def get_actions(self):
        return self.actions_done

    def get_num_actions(self):
        return self.last_piece_num_actions

    def render_frame(self, canvas, cell_size):

        main_grid = pygame.Surface((cell_size * 10 + 1, cell_size * 20 + 1))
        main_grid.fill((0, 0, 0))
        reserve_grid = pygame.Surface((cell_size * 2 + 1, cell_size * 1 + 1))
        reserve_grid.fill((0, 0, 0))
        next_grid = pygame.Surface((cell_size * 2 + 1, cell_size * 9 + 1))
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
            tetromino: tetromino to draw on surface.
            surface: surface to draw tetromino onto.
            cell_size: size in pixels of single square of a tetromino.
            main_grid: bool to know if drawing tetromino on main grid or reserved/next grid.
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
        render_tetr(self.current_tetr, main_grid, cell_size)

        # Render played tetrominoes
        if self.played_tetr:
            for tetr in self.played_tetr:
                render_tetr(tetr, main_grid, cell_size)

        # Render reserved tetromino
        if self.reserved_tetr:
            render_tetr(self.reserved_tetr, reserve_grid, cell_size//2, False)

        # Render next tetrominoes
        for i, tetr in enumerate(self.next_tetr):
            off_set = 3*i * cell_size//2
            render_tetr(tetr, next_grid, cell_size//2, False, (0, off_set))


        # Finally, add some gridlines
        draw_grid_lines(main_grid, cell_size)
        draw_grid_lines(reserve_grid, cell_size//2)
        draw_grid_lines(next_grid, cell_size//2)

        # Blits the different surfaces onto canvas
        canvas.blit(source=main_grid, dest=(4*cell_size, 2*cell_size))
        canvas.blit(source=reserve_grid, dest=(1*cell_size, 2*cell_size))
        canvas.blit(source=next_grid, dest=(15*cell_size, 2*cell_size))

        return canvas


class BlockID(Enum):
    I = 1
    O = 2
    L = 3
    J = 4
    S = 5
    Z = 6
    T = 7


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
        if name == BlockID.J:
            return JBlock()
        elif name == BlockID.S:
            return SBlock()
        elif name == BlockID.I:
            return IBlock()
        elif name == BlockID.L:
            return LBlock()
        elif name == BlockID.Z:
            return ZBlock()
        elif name == BlockID.O:
            return OBlock()
        elif name == BlockID.T:
            return TBlock()

        # If none type specified returns a random tetromino
        pool = (IBlock(), LBlock(), JBlock(), SBlock(), ZBlock(), OBlock(), TBlock())
        tetromino = random.choice(pool)
        return tetromino

    # @staticmethod
    # def remove_empty_rows(struct_array):
    #     new_array = np.copy(struct_array)
    #     rows, _ = np.where(new_array == 1)
    #     y_offset = min(rows)
    #     new_array = new_array[~np.all(new_array == 0, axis=1)]
    #     return new_array, y_offset

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
    name = BlockID.I
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
    name = BlockID.L
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
    name = BlockID.J
    rotations = 4
    _x = 3
    _y = -1


class OBlock(Tetromino):
    struct = np.array(
        ((1, 1),
         (1, 1))
    )
    color = (222, 165, 0)
    name = BlockID.O
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
    name = BlockID.S
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
    name = BlockID.T
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
    name = BlockID.Z
    rotations = 4
    _x = 3
    _y = -1


# Piece not in use
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


''' Plan movement based on tetris state
INPUT:
    - int: column of the board (x-position i.e. top-left cell x-position)
    - int: rotation state of the piece

OUTPUT:
    - list of movements to achieve desired position

INITIAL APPROACH 
    - rotate piece to desired rotation of the piece
    - move along x axis
    - drop piece
    ASSUMPTIONS MADE:
    - Piece will not be blocked when droped from any y-position and will get to the desired y-position
    - Piece will not perform a wall-kick in the initial rotation

'''


def movement_planning(tetr: Tetromino, x_pos: int, rot_state: int):
    movements = []
    rotations = rot_state - tetr._rotation_state
    movements.extend([Action.ROT_RIGHT] * rotations)
    position = x_pos - tetr._x
    if position > 0:
        movements.extend([Action.RIGHT] * position)
    elif position < 0:
        movements.extend([Action.LEFT] * np.abs(position))
    movements.append(Action.DROP)
    return movements

class MyGUI:
    def __init__(self):
        self.state = TetrisState()
        self.cell_size = 32
        self.height = 23
        self.width = 18
        self.window_size = np.array((self.width * self.cell_size, self.height * self.cell_size))
        self.game_over = False
        self.action = None

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(self.window_size)
        self.canvas = pygame.Surface(self.window_size)

    def run(self):
        while not self.game_over:
            self.process_input()
            self.update()
            self.render()

    def process_input(self):
        self.action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.game_over = True
                elif event.key == pygame.K_a:
                    self.action = Action.LEFT.value
                elif event.key == pygame.K_d:
                    self.action = Action.RIGHT.value
                elif event.key == pygame.K_s:
                    self.action = Action.DOWN.value
                elif event.key == pygame.K_w:
                    self.action = Action.DROP.value
                elif event.key == pygame.K_j:
                    self.action = Action.ROT_LEFT
                elif event.key == pygame.K_k:
                    self.action = Action.ROT_RIGHT

    def update(self):
        x_pos = np.random.choice(range(10))
        rot_state = np.random.choice(range(4))
        print(f'Position: {x_pos}, rotation: {rot_state}')
        movements = movement_planning(self.state.current_tetr, x_pos, rot_state)
        print(movements)
        for move in movements:
            self.game_over, _, _ = self.state.update(move.value)

    def render(self):
        self.canvas.fill((0, 0, 0))
        self.canvas = self.state.render_frame(self.canvas, self.cell_size)
        self.window.blit(self.canvas, self.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()



if __name__ == '__main__':
    game = MyGUI()
    game.run()
    pygame.quit()
