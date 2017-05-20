import cv2
import copy
import os 
from train import train_knn
from utils import get_sudoku_board, extract_number, predict
from solve import solve_sudoku, has_duplicate, print_board

def rotate(sudoku_board):
    rows, cols = sudoku_board.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
    rotated_board = cv2.warpAffine(sudoku_board, M, (rows, cols))
    return rotated_board

def solve(numbers, knn):
    grid = [[0]*9 for _ in range(9)]
    for number, (x,y) in numbers:
        label = predict(number, knn)
        label = int(label)
        grid[y//47][x//47] = label
    print_board(grid)
    original_grid = copy.deepcopy(grid)
    if has_duplicate(grid):
        return (False, None, None) 
    else:
        if solve_sudoku(grid, 0, 0):
            return True, grid, original_grid
    return False, None, None

def solve_image_sudoku(path):
    board = get_sudoku_board(path)
    numbers = extract_number(copy.copy(board))
    # cv2.imshow('board', board)
    knn = train_knn()
    for _ in range(5):
        if len(numbers) > 0:
            cv2.imshow('board', board)
            can_solve, grid, orgi_grid = solve(numbers, knn)
            if can_solve:
                draw_back(grid, orgi_grid, board)
                cv2.imshow('solved_board', board)
                print_board(grid)
                print(grid == orgi_grid)
                return True 
        board = rotate(board)
        numbers = extract_number(copy.copy(board))
    else:
        print("Cant solve")
        return False

def draw_back(grid, original_grid, board):
    for i in range(9):
        for j in range(9):
            if original_grid[i][j] == 0:
                cv2.putText(
                    board, 
                    str(grid[i][j]), 
                    ((j)*47+15, (i+1)*47-13), 
                    0, 1, (0, 0, 255), 2 
                )

if __name__ == '__main__':
    TEST_FOLDER = '../images/test-images'
    success, failed = 0, 0
    for file_name in os.listdir(TEST_FOLDER):
        print(file_name)
        file_path = os.path.join(TEST_FOLDER, file_name)
        if solve_image_sudoku(file_path):
            success += 1
        else:
            failed += 1
        cv2.waitKey()
    print("Sucess: {}\nFailed: {}".format(success, failed))
    
    # while True:
    #     if cv2.waitKey(1) == 27:
    #         break
    # for number, (x,y) in numbers:
    #     cv2.imshow('number', number)
    #     label = predict(number, knn)
    #     print(label)
    #     key = cv2.waitKey()
    #     if key == 27:
    #         break