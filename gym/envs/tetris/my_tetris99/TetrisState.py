from enum import Enum
import numpy as np
from Tetromino import Tetromino
import time
import pygame





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

    def reset(self):
        self.__init__()


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

    def movement_planning(self, x_pos: int, rot_state: int):
        ghost_tetr = Tetromino.make(self.current_tetr.name)
        movements = []
        rotations = rot_state - self.current_tetr._rotation_state
        movements.extend([Action.ROT_RIGHT] * rotations)
        ghost_tetr.rotate(-rotations, None)
        position = x_pos - ghost_tetr.left
        if position > 0:
            movements.extend([Action.RIGHT] * position)
        elif position < 0:
            movements.extend([Action.LEFT] * np.abs(position))
        movements.append(Action.DROP)
        return movements
