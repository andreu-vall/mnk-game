import numpy as np
import mnk_gravity_game
import pygame

m, n, k = 7, 6, 4
max_time = 1
pc_play, pc_start = True, True

square_size = np.array([100, 100])
board_size = np.array([m, n+1])
game_fps = 24

screen_size = square_size * board_size

pygame.init()
running = True
pygame.display.set_caption(f"{m},{n},{k}-gravity game")

screen = pygame.display.set_mode(screen_size)


def clean_top(color):
    rectangle = pygame.Rect([(0, 0), (screen_size[0], square_size[1])])
    screen.fill(pygame.Color(color), rectangle)
    pygame.display.update(rectangle)


def draw_board(board_color, screen_color):
    screen.fill(pygame.Color(board_color))
    clean_top(screen_color)
    for pos in np.array(np.meshgrid(range(board_size[0]), range(1, board_size[1]))).T.reshape(-1, 2):
        pygame.draw.circle(screen, pygame.Color(screen_color), pos*square_size + square_size//2, 45)
    pygame.display.flip()


def draw_circle(circle_color, pos):
    pygame.draw.circle(screen, pygame.Color(circle_color), pos*square_size + square_size//2, 45)
    pygame.display.update((pos*square_size, square_size))


def draw(turn, position):
    if turn == 1:
        draw_circle("red", position)
    else:
        draw_circle("yellow", position)


draw_board("blue", "black")

game = mnk_gravity_game.MnkGravityGame(m, n, k)


def pc_move():
    move = game.iterative_deepening_search(max_time)
    game.put(move)
    draw(-game.turn, [move[1], move[0]+1])


if pc_play and pc_start:
    pc_move()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYUP and event.key == pygame.K_r:
            game.restart()
            draw_board("blue", "black")
            if pc_play and pc_start:
                pc_move()

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_position = pygame.mouse.get_pos()//square_size
            row = int(mouse_position[0])
            real_position = np.array([row, game.positions[row]+1])

            if game.put(row):
                draw(-game.turn, real_position)
                if pc_play and not game.finished:
                    pc_move()

    clean_top("black")
    mouse_position = np.array([pygame.mouse.get_pos()[0]//square_size[0], 0])
    draw(game.turn, mouse_position)

    pygame.time.Clock().tick(game_fps)

pygame.quit()
