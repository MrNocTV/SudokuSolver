def find_next_cell(grid, i, j):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return i,j
    # return tuple 
    return -1, -1

def is_valid(grid, i, j, value):
    valid_row = all([value != grid[i][x] for x in range(9)])
    if valid_row:
        valid_column = all([value != grid[x][j] for x in range(9)])
        if valid_column:
            top_x = 3*(i//3)
            top_y = 3*(j//3)
            for x in range(top_x, top_x+3):
                for y in range(top_y, top_y+3):
                    if grid[x][y] == value:
                        return False 
            return True 
    return False 

def solve_sudoku(grid, i, j):
    i, j = find_next_cell(grid, i, j)
    if i == -1:
        return True 
    for value in range(1, 10): 
        if is_valid(grid, i, j, value):
            grid[i][j] = value 
            if solve_sudoku(grid, i, j):
                return True 
            grid[i][j] = 0
    return False 

def has_duplicate(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                valid_row = all(grid[i][j] != grid[i][x] for x in range(9) if x != j)
                if valid_row:
                    valid_column = all([grid[i][j] != grid[x][j] for x in range(9) if x != i])
                    if valid_column:
                        top_x = 3*(i//3)
                        top_y = 3*(j//3)
                        for x in range(top_x, top_x+3):
                            for y in range(top_y, top_y+3):
                                if x != i and j != y:
                                    if grid[x][y] == grid[i][j]:
                                        return True
                    else:
                        return True
                else:
                    return True
    return False 

def print_board(grid):
    print("="*21)
    for i in range(0, len(grid)):
        for j in range(0, len(grid[0])):
            if j % 3 == 0:
                print(" ", end='')
            print('{} '.format(grid[i][j]), end='')
        print()
        if (i+1) % 3 == 0:
            print()
    print("="*21 + '\n')
