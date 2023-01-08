import random
from collections import deque

import numpy as np
import pygame

BASE = 0
ROW_NUMBER = 4
COL_NUMBER = 4
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
RED_BASE = 255

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 50
HEIGHT = 50

# This sets the margin between each cell
MARGIN = 5

class Board:
    def __init__(self, render, output):
        self.matrix = np.array([[BASE] * COL_NUMBER for i in range(ROW_NUMBER)])
        #self.matrix = np.array([[ 2 ,4 ,2, 4],[ 8,  256,  4,  8],[ 2,  0,  256,  4],[0,  4,  2,  4]])
        self.score = 0
        self.max = 0
        if render:
            pygame.init()

            # Set the HEIGHT and WIDTH of the screen
            WINDOW_SIZE = [225, 225]
            screen = pygame.display.set_mode(WINDOW_SIZE)
            screen.fill(RED)

            # Set title of screen
            pygame.display.set_caption("2048")

            # Loop until the user clicks the close button.
            done = False

            # Used to manage how fast the screen updates
            clock = pygame.time.Clock()
            self.pygame = pygame
            self.screen = screen
            self.clock = clock

        self.render = render
        self.output = output

        self.generate_2()
        self.display()
        self.is_game_over()


    def generate_2(self):
        temp = []
        for i in range(ROW_NUMBER):
            for j in range(COL_NUMBER):
                if self.matrix[i][j] == BASE:
                    temp.append((i, j))
        loc_x, loc_y = temp[random.randint(0, len(temp) - 1)]
        self.matrix[loc_x][loc_y] = 2
        self.is_game_over()

    def is_merge_possible(self):
        for row_index in range(ROW_NUMBER):
            row = self.matrix[row_index]
            for col_index in range(COL_NUMBER - 1):
                if row[col_index] == row[col_index + 1]:
                    return True
        for col_index in range(COL_NUMBER):
            column = self.matrix[:,col_index]
            for row_index in range(ROW_NUMBER - 1):
                if column[row_index] == column[row_index + 1]:
                    return True
        return False

    def is_game_over(self):
        if np.count_nonzero(self.matrix == BASE) == 0 and not self.is_merge_possible():
            if self.output:
                print(f'You lose with score: {self.score} and max {self.max}')
                print(f'Press space to reset')
            return True
        else:
            return False

    def merge_left(self):

        reward = 0
        for i in range(4):
            row = self.matrix[i].copy()
            queue = deque(row)
            new_row = []
            while queue:
                cur = deque.popleft(queue)
                if cur == BASE:
                    new_row.append(cur)
                else:
                    if queue:
                        if cur == queue[0]:
                            new_row.append(cur * 2)
                            self.score += cur * 2
                            self.max = max(self.max, cur * 2)
                            deque.popleft(queue)
                            reward += cur * 2
                            #print(f'merge{cur}')
                        else:
                            new_row.append(cur)
                    else:
                        new_row.append(cur)
            while len(new_row) != 4:
                new_row.append(BASE)
            self.matrix[i] = new_row
        return 15 if reward != 0 else 0

    def move_left_base(self):
        moved = False
        for i in range(4):
            row = self.matrix[i].copy()
            new_row = [x for x in row if x != BASE]
            while len(new_row) != 4:
                new_row.append(BASE)
            self.matrix[i] = new_row
            if list(new_row) != list(row):
                moved = True
        return moved

    def move_left(self):
        moved = self.move_left_base()
        reward = self.merge_left()
        return self.get_final_reward(reward, moved)

    def move_right(self):
        self.matrix = np.flip(self.matrix, axis=1)
        moved = self.move_left_base()
        reward = self.merge_left()
        self.matrix = np.flip(self.matrix, axis=1)
        #self.process_response(moved, merged)
        return self.get_final_reward(reward, moved)

    def move_up(self):
        self.matrix = np.rot90(self.matrix)
        moved = self.move_left_base()
        reward = self.merge_left()
        self.matrix = np.rot90(self.matrix, -1)
        #self.process_response(moved, merged)
        return self.get_final_reward(reward, moved)

    def move_down(self):
        self.matrix = np.rot90(self.matrix, -1)
        moved = self.move_left_base()
        reward = self.merge_left()
        self.matrix = np.rot90(self.matrix)
        #self.process_response(moved, merged)
        return self.get_final_reward(reward, moved)

    def display(self):
        if self.render:
            for row in range(4):
                for column in range(4):
                    red = max(RED_BASE - RED_BASE * (self.matrix[row][column] / 2048), 0)
                    color = (255, red, red)
                    self.pygame.draw.rect(self.screen,
                                     color,
                                     [(MARGIN + WIDTH) * column + MARGIN,
                                      (MARGIN + HEIGHT) * row + MARGIN,
                                      WIDTH,
                                      HEIGHT])
                    if self.matrix[row][column] != 0:
                        font = self.pygame.font.Font(None, 25)
                        text = font.render(str(self.matrix[row][column]), True, BLACK)
                        text_rect = text.get_rect(center=(
                        (MARGIN + WIDTH) * column + MARGIN + 0.5 * WIDTH, (MARGIN + HEIGHT) * row + MARGIN + 0.5 * HEIGHT))
                        self.screen.blit(text, text_rect)

            # Limit to 60 frames per second
            self.clock.tick(3000)

            # Go ahead and update the screen with what we've drawn.
            self.pygame.display.flip()
        else:
            if self.output:
                print(self.matrix)
            pass

    def process_response(self, moved, merged):
        if not moved and not merged:
            #print("Invalid Move")
            return False
        else:
            self.is_game_over()
            self.generate_2()
            self.display()
            return True

    def reset(self):
        self.matrix = np.array([[BASE] * 4 for i in range(4)])
        #self.matrix = np.array([[ 2 ,4 ,2, 4],[ 8,  256,  4,  8],[ 2,  0,  256,  4],[0,  4,  2,  4]])
        self.score = 0
        self.max = 0
        self.generate_2()
        self.display()

    def get_final_reward(self, merge_reward, moved):
        if self.output:
            print(f'cur: max{self.max} score: {self.score}')
        if not moved and not merge_reward:
            #print("Invalid Move")
            if self.output:
                print(f'reward -5')
            return -5
        elif moved and not merge_reward:
            if self.is_game_over():
                if self.output:
                    print(f'reward -15')
                return -15
            self.generate_2()
            self.display()
            if self.output:
                print(f'reward 0')
            return 0
        else:
            if self.is_game_over():
                if self.output:
                    print(f'reward -15')
                return -15
            self.generate_2()
            self.display()
            if self.output:
                print(f'reward {merge_reward}')
            return merge_reward

