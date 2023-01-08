import datetime
from pathlib import Path
from random import randrange

import numpy as np
from matplotlib import pyplot as plt

from pynput import keyboard
from pynput.keyboard import Key, Events

from agent import Agent
from board import Board
import matplotlib

MODE = 1 # 0 for human 1 for rl training
DISPLAY_PLOT = 1
RENDER_UI = 0
OUTPUT_DETAILS = 0
episodes = 20000
save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)]

checkpoint = None
record = np.array([])

ep_temp_sum = 0
batch_size = 20


def main():
    global ep_temp_sum
    """
    input = torch.randn(1, 1, 4, 4)
    # With default parameters
    m = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=1, padding=1),
        #nn.Conv2d(in_channels=c, out_channels=6, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        #nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1),
        #nn.ReLU(),
        #nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1),
        #nn.ReLU(),
        nn.Flatten(),
        nn.Linear(72, 9),
        nn.ReLU(),
        nn.Linear(9, 4)

    )
    output = m(input)
    print(output.size())
    exit()
    """
    if DISPLAY_PLOT:
        matplotlib.use("TkAgg")
        ep = np.array([])
        score_list = np.array([])
        avg_score_list = np.array([])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(ep, score_list, 'r-')
        line2, = ax.plot(ep, avg_score_list, 'g-')
        plt.draw()

    """
        def update_points(new_x, new_y):
            nonlocal ep, score_list
            ep = np.append(ep, new_x)
            score_list = np.append(score_list, new_y)
            line1.set_data(ep, score_list)
            plt.draw()
            plt.pause(0.02)
    """
    board = Board(render=RENDER_UI, output=OUTPUT_DETAILS)
    agent = Agent(state_dim=(1, 4, 4), action_dim=4, save_dir=save_dir, checkpoint=checkpoint)
    best_score_matrix = np.array([])
    best_score = 0

    if not MODE:
        with keyboard.Events() as events:
            for event in events:
                if type(event) == Events.Press:
                    key = event.key
                    if key == Key.right:
                        board.move_right()
                    elif key == Key.left:
                        board.move_left()
                    elif key == Key.up:
                        board.move_up()
                    elif key == Key.down:
                        board.move_down()
                    elif key == Key.space:
                        board.reset()
                    elif key == Key.esc:
                        exit()
    else:
        for e in range(episodes):
            id_list = []
            while True:
                # get state
                temp_matrix = board.matrix.copy()
                # state = temp_matrix.reshape((4,4,1))
                # get action
                idx = agent.act(temp_matrix)
                id_list.append(idx)
                #print(idx)
                action = action_space[idx]
                #action = action_space[randrange(4)]
                #
                reward = 0
                if action == (-1, 0):
                    #print("left")
                    reward = board.move_left()
                elif action == (1, 0):
                    #print("right")
                    reward= board.move_right()
                elif action == (0, -1):
                    #print("up")
                    reward = board.move_up()
                elif action == (0, 1):
                    #print("down")
                    reward = board.move_down()

                next_temp_matrix = board.matrix.copy()
                # next_state = temp_matrix.reshape((4, 4, 1))
                score = board.score
                done = board.is_game_over()

                if score >= best_score:
                    best_score = score
                    best_score_matrix = next_temp_matrix.copy()
                #print(f'idx:{idx} reward: {reward} done {done}')
                agent.cache(temp_matrix, next_temp_matrix, idx, reward, done)

                # learn
                q, loss = agent.learn()

                # logging
                #print(f'score: {score}, loss: {loss}, q: {q}')

                if done:
                    break



            # update_points(np.array([e]), np.array([score]))

            if DISPLAY_PLOT:
                ep = np.append(ep, e + 1)
                #score_list = np.append(score_list, board.max)
                score_list = np.append(score_list, board.score)
                avg = sum(score_list) / (e + 1)
                avg_score_list = np.append(avg_score_list, avg)
                line1.set_data(ep, score_list)
                line2.set_data(ep, avg_score_list)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.000002)
                if OUTPUT_DETAILS:
                    print(f'episode {e} complete with max: {board.max} score: {board.score}')

            board.reset()

    print("complete")
    print(f'best score: {best_score} with matrix {best_score_matrix}')
    if DISPLAY_PLOT:
        plt.show()


if __name__ == "__main__":
    main()
