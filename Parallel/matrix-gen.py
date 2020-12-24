import random


def generate_matrix(dim):
    matrixA = []
    for row_num in range(0, dim):
        num_list = []
        for col_num in range(0, dim):
            if row_num == col_num:
                num_list.append(random.randint(dim * 100 + 1, ((dim * 100 + 1) * 10)))
            else:
                num_list.append(random.randint(-100, 100))
        matrixA.append(num_list)

    f = open("matrixA.txt", "w")
    for row in matrixA:
        f.write(" ".join([str(int) for int in row]))
        f.write("\n")
    f.close()

    matrixB = []
    for elem in range(0, dim):
        matrixB.append(random.randint(-100, 100))

    f = open("matrixB.txt", "w")
    for elem in matrixB:
        f.write(str(elem) + "\n")
    f.close()


generate_matrix(1000)

