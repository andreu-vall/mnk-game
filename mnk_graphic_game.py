import numpy as np
import mnk_game
import pygame

m, n, k = 5, 5, 4
max_time = 1
pc_play, pc_start = True, True

square_size = np.array([100, 100])
board_size = np.array([m, n])
margin_size = np.array([10, 10])
game_fps = 10

screen_size = square_size * board_size + margin_size * (board_size - 1)

pygame.init()
running = True
pygame.display.set_caption(f"{m},{n},{k}-game")

screen = pygame.display.set_mode(screen_size)


def draw_board(board_color, screen_color):
    screen.fill(pygame.Color(screen_color))
    for i in range(1, board_size[0]):
        pygame.draw.rect(screen, pygame.Color(board_color),
                         (i*square_size[0] + (i-1)*margin_size[0], 0, margin_size[0], screen_size[1]))
    for j in range(1, board_size[1]):
        pygame.draw.rect(screen, pygame.Color(board_color),
                         (0, j*square_size[1] + (j-1)*margin_size[1], screen_size[0], margin_size[1]))
    pygame.display.flip()


def draw_circle(circle_color, position):
    center = (square_size*(position+1/2) + margin_size*position).astype(int)
    pygame.draw.circle(screen, pygame.Color(circle_color), center, np.round(square_size[0]*0.45).astype(int), 5)
    pygame.display.update((center - square_size//2, square_size))


def draw_cross(cross_color, position):
    top_left = position * (square_size+margin_size)
    top_right = top_left + square_size * np.array([1, 0])
    space = np.array([10, 10])
    gir = np.array([-1, 1])
    pygame.draw.line(screen, pygame.Color(cross_color), top_left+space, top_left+square_size-space, 5)
    pygame.draw.line(screen, pygame.Color(cross_color), top_right + space*gir, top_right + (square_size-space)*gir, 5)
    pygame.display.update((top_left, square_size))


def draw_turn(player, position):
    if player == 1:
        draw_cross("red", position)
    else:
        draw_circle("blue", position)


draw_board("black", "white")

game = mnk_game.MnkGame(m, n, k)


def pc_move():
    move = game.iterative_deepening_search(max_time)
    game.put(move)
    draw_turn(-game.turn, np.array(move[::-1]))


if pc_play and pc_start:
    pc_move()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYUP and event.key == pygame.K_r:
            game.restart()
            draw_board("black", "white")
            if pc_play and pc_start:
                pc_move()

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            coord = np.array(pygame.mouse.get_pos())
            mouse_position = (coord + margin_size // 2) // (square_size+margin_size)
            tuple_pos = tuple(mouse_position[::-1])

            if game.put(tuple_pos):
                draw_turn(-game.turn, mouse_position)
                if pc_play:
                    if not game.finished:
                        pc_move()
                else:
                    print(game.finished, game.won, game.heuristic)

    pygame.time.Clock().tick(game_fps)

pygame.quit()
