from random import randint

def two_dimensional_random_walk():
    steps = 0
    # Steps counter for understand how many steps that our drunken man take
    grid_size = 11
    # creating two dimensional array using lists
    times = [0] * grid_size

    for i in range(0, grid_size):
        times[i] = [0] * grid_size

    # initial variables to start in the middle of grid
    x = 5
    y = 5

    # Tuples to get directions and decide where to go
    moves = [(1,0,"right"), (0,1,"up"), (-1,0,"left"), (0,1,"down")]

    # My loop for evaluate the steps
    while True:
        dx, dy, position = moves[randint(0,3)] # by using randint I could make decision randomly
        x += dx
        y += dy
        print("He moved", position)

        try:
            times[x][y] += 1 # And here is, how many times have he stood on each square
            steps += 1
        except IndexError: # The exit of loop
            break

    # My print function which answers these questions (How long will it be until he reaeches the end of the sidewalk, and how many times will he have stood on each square)
    for i in range(0,11):
        for j in range(0,11):
            print("He took {steps} steps until he reaches end of the sidewalk.".format(steps = steps),  "He stood on {1}x{2} square at {0} times".format(times[i][j],i+1,j+1))


