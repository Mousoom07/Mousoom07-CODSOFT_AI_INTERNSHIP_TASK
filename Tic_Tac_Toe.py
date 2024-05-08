# Function to print the board
def print_board(board):
    for row in board:
        print(" ".join(row))

# Function to check if a player has won
def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or (all(board[i][2-i] == player for i in range(3))):
        return True
    return False

# Function to check if the board is full
def is_board_full(board):
    return all(cell != ' ' for row in board for cell in row)

# Function to get a human player's move
def get_human_move(board):
    while True:
        try:
            row = int(input("Enter the row (0, 1, or 2): "))
            col = int(input("Enter the column (0, 1, or 2): "))
            if board[row][col] == ' ':
                return row, col
            else:
                print("That position is already taken! Try again.")
        except ValueError:
            print("Invalid input! Please enter a number.")

# Function for the AI to make a move
def get_ai_move(board):
    for row in range(3):
        for col in range(3):
            if board[row][col] == ' ':
                return row, col

# Main game loop
def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = 'X'

    while True:
        print_board(board)

        if check_winner(board, 'X'):
            print("X wins!")
            break
        elif check_winner(board, 'O'):
            print("O wins!")
            break
        elif is_board_full(board):
            print("It's a tie!")
            break

        if current_player == 'X':
            row, col = get_human_move(board)
        else:
            row, col = get_ai_move(board)

        board[row][col] = current_player
        current_player = 'O' if current_player == 'X' else 'X'

# Start the game
play_game()