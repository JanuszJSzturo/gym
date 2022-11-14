import pygame
from TetrisState import TetrisState, Action
import numpy as np


class MyGUI:
    def __init__(self):
        self.state = TetrisState()
        self.cell_size = 32
        self.height = 23
        self.width = 18
        self.window_size = np.array((self.width * self.cell_size, self.height * self.cell_size))
        self.game_over = False
        self.quit = False
        self.action = None
        self.clock = pygame.time.Clock()
        self.next = True

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(self.window_size)
        self.canvas = pygame.Surface(self.window_size)

    def run(self):
        while not self.game_over and not self.quit:
            self.process_input()
            self.update_human()
            if self.next:
                self.update()
            self.render()

    def process_input(self):
        self.action = None
        self.next = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit = True
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
                elif event.key == pygame.K_SPACE:
                    self.next = True

    def update(self):
        x_pos = np.random.choice(range(10))
        rot_state = np.random.choice(range(4))
        print(f'Start position: {self.state.current_tetr._x}, start rotation: {self.state.current_tetr._rotation_state}')
        print(f'End Position: {x_pos}, end   rotation: {rot_state}')
        movements = self.state.movement_planning(x_pos, rot_state)
        print(movements)
        for move in movements:
            self.game_over, _, _ = self.state.update(move.value)

    def update_human(self):
        self.game_over, _, _ = self.state.update(action=self.action)

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

