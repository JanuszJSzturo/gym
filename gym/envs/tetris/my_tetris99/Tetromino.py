from enum import Enum
import random
import numpy as np
import copy

WALL_KICKS_JLTSZ = ([[[(-1, 0), (-1, 1), (0, -2), (-1, -2)], [(1, 0), (1, 1), (0, -2), (1, -2)]],
                     [[(1, 0), (1, -1), (0, 2), (1, 2)], [(1, 0), (1, -1), (0, 2), (1, 2)]],
                     [[(1, 0), (1, 1), (0, -2), (1, -2)], [(-1, 0), (-1, 1), (0, -2), (-1, -2)]],
                     [[(-1, 0), (-1, -1), (0, 2), (-1, 2)], [(-1, 0), (-1, -1), (0, 2), (-1, 2)]]])

WALL_KICKS_I = ([[[(-2, 0), (1, 0), (-2, -1), (1, 2)], [(-1, 0), (2, 0), (-1, 2), (2, -1)]],
                 [[(-1, 0), (2, 0), (-1, 2), (2, -1)], [(2, 0), (-1, 0), (2, 1), (-1, -2)]],
                 [[(2, 0), (-1, 0), (2, 1), (-1, -2)], [(1, 0), (-2, 0), (1, -2), (-2, 1)]],
                 [[(1, 0), (-2, 0), (1, -2), (-2, 1)], [(-2, 0), (1, 0), (-2, -1), (1, 2)]]])


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
    def make(name: BlockID = None):
        """
        Returns a tetromino of type "name" or random tetromino if name is not specified o does not exist
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

        # If type specified returns a random tetromino
        tetromino = random.choice(POOL)()
        return tetromino

    # @staticmethod
    # def remove_empty_rows(struct_array):
    #     new_array = np.copy(struct_array)
    #     rows, _ = np.where(new_array == 1)
    #     y_offset = min(rows)
    #     new_array = new_array[~np.all(new_array == 0, axis=1)]
    #     return new_array, y_offset

    @property
    def left(self):
        _, cols = np.where(self.struct == 1)
        return self._x + min(cols)

    @property
    def right(self):
        _, cols = np.where(self.struct == 1)
        return self._x + max(cols)

    @property
    def top(self):
        rows, _ = np.where(self.struct == 1)
        return self._y + min(rows)

    @property
    def bot(self):
        rows, _ = np.where(self.struct == 1)
        return self._y + max(rows)

    def collision(self, others):
        """
        Return True if a tetromino collides with other in others, else return False
        """
        if self.left < 0 or self.right > 9 or self.bot > 19:
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
        self.struct = np.rot90(self.struct, k=rot_direction)  # np.rot90 k>0 counter-clockwise, k<0 clockwise
        self._rotation_state = (self._rotation_state - rot_direction) % 4
        if collision_group is None:
            return
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


POOL = (IBlock, LBlock, JBlock, SBlock, ZBlock, OBlock, TBlock)

if __name__ == '__main__':
    print(f'running tetrmino')