import copy
import random
from collections import deque
import math
import time

class Matrix():
    def __init__(self, rows=0, columns=0, matrix = []):
        self.matrix = matrix
        self.rows = rows
        self.columns = columns

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.matrix])

    def __repr__(self):
        return '\n\n['+']\n['.join([', '.join(map(str, row)) for row in self.matrix])+']'

    def __add__(self, other):
        if self.rows != other.rows:
            raise ValueError("row/column 다름: {} != {}".format(self.rows, other.rows))
        if self.columns != other.columns:
            raise ValueError("row/column 다름: {} != {}".format(self.columns, other.columns))
        result = Matrix()
        result.rows = self.rows
        result.columns = self.columns
        result.matrix = [[self.matrix[i][j] + other.matrix[i][j] for j in range(self.columns)] for i in range(self.rows)]
        return result

    def __sub__(self, other):
        if self.rows != other.rows:
            raise ValueError("row/column 다름: {} != {}".format(self.rows, other.rows))
        if self.columns != other.columns:
            raise ValueError("row/column 다름: {} != {}".format(self.columns, other.columns))
        result = Matrix()
        result.rows = self.rows
        result.columns = self.columns
        result.matrix = [[self.matrix[i][j] - other.matrix[i][j] for j in range(self.columns)] for i in range(self.rows)]
        return result

    def __mul__(self, other):
        if self.columns != other.rows:
            raise ValueError("row/column 다름: {} != {}".format(self.columns, other.rows))
        result = Matrix()
        result.rows = self.rows
        result.columns = other.columns
        result.matrix = [[0] * other.columns for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.columns):
                for k in range(self.columns):
                    result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return result

    def __eq__(self, other):
        if self.rows != other.rows:
            return False
        if self.columns != other.columns:
            return False
        for i in range(self.rows):
            for j in range(self.columns):
                if self.matrix[i][j] != other.matrix[i][j]:
                    return False
        return True

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.matrix))

    def transpose(self):
        result = Matrix()
        result.rows = self.columns
        result.columns = self.rows
        result.matrix = [[self.matrix[j][i] for j in range(self.rows)] for i in range(self.columns)]
        return result

    def gaussian_elimination(self):
        pivot_row = 0
        for pivot_col in range(self.columns):
            q = pivot_row
            for i in range(pivot_row+1, self.rows):
                if abs(self.matrix[i][pivot_col]) > abs(self.matrix[q][pivot_col]):
                    q = i
            for j in range(self.columns):
                self.matrix[pivot_row][j], self.matrix[q][j] = self.matrix[q][j], self.matrix[pivot_row][j]
            if self.matrix[pivot_row][pivot_col] == 0:
                continue
            for i in range(pivot_row+1, self.rows):
                c = self.matrix[i][pivot_col] / self.matrix[pivot_row][pivot_col]
                for j in range(pivot_col, self.columns):
                    self.matrix[i][j] -= c * self.matrix[pivot_row][j]
            pivot_row += 1

    def det(self):
        if self.rows != self.columns:
            raise ValueError("square matrix 아님: {} != {}".format(self.rows, self.columns))
        augmented = Matrix()
        augmented.rows = self.rows
        augmented.columns = self.columns
        augmented.matrix = self.matrix
        augmented.gaussian_elimination()
        det = 1
        for i in range(self.rows):
            det *= augmented.matrix[i][i]
        return det

    def inverse(self):
        if self.rows != self.columns:
            raise ValueError("square matrix 아님: {} != {}".format(self.rows, self.columns))
        augmented = Matrix()
        augmented.rows = self.rows
        augmented.columns = 2 * self.columns
        augmented.matrix = [row + [0] * self.columns for row in self.matrix]
        for i in range(self.rows):
            augmented.matrix[i][i + self.columns] = 1
        augmented.gaussian_elimination()
        for i in range(self.rows):
            if augmented.matrix[i][i] == 0:
                raise ValueError("역행렬 없음")
        inverse = Matrix()
        inverse.rows = self.rows
        inverse.columns = self.columns
        inverse.matrix = [[augmented.matrix[i][j + self.columns] for j in range(self.columns)] for i in range(self.rows)]
        return inverse

#matrix class로 바꾸기

def cromwell_binary(cromwell):
    n = len(cromwell)
    binary_list = []
    for i in range(n):
        a=0
        for j in range(n+1):
            a += cromwell[i][j] * (2**(n+1-j))
        binary_list.append(a)
    sum = 0
    for i in range(len(binary_list)):
        sum += binary_list[i]
    return sum
print(cromwell_binary([[1,0,1],[0,0,1]]))


#cromwell matrix를 모은 list 같은 게 필요함, 나중에 수정하기, 현재는 matrix 하나를 넣으면 symmetry인 것만 리턴함
cromwell_list = []
def symmetry_delete(cromwell):
    n = len(cromwell)
    symmetry_delete_list = []

    symmetry_toprow = copy.deepcopy(cromwell)
    for i in range(n):
        for j in range(n+1):
            symmetry_toprow[i][j] = cromwell[n-i-1][j]
    symmetry_delete_list.append(symmetry_toprow)

    symmetry_leftrow = copy.deepcopy(cromwell)
    for i in range(n):
        for j in range(n+1):
            symmetry_leftrow[i][j] = cromwell[i][n-j]
    symmetry_delete_list.append(symmetry_leftrow)


    symmetry_topleftrow = copy.deepcopy(cromwell)
    for i in range(n):
        for j in range(n+1):
            symmetry_topleftrow[i][j] = cromwell[n-i-1][n-j]
    symmetry_delete_list.append(symmetry_topleftrow)

    return symmetry_delete_list

print(symmetry_delete([[1,1,1,0],[0,1,1,1],[1,0,0,1]]))

def VerticalOne(cromwell):
    n = cromwell.rows
    three = []
    for row_i in range(n):
            if sum(cromwell.matrix[row_i]) == 3:
                three.append(row_i)
    for i in range(n-1):
        for j in range(n+1):
            if cromwell.matrix[i][j] == cromwell.matrix[i+1][j] == 1 and i not in three and i+1 not in three:
                return True
    return False

def HorizontalOne(cromwell):
    n = cromwell.rows
    m = cromwell.columns

    # 3개짜리 제외
    three = []
    for row_i in range(n):
        if sum(cromwell.matrix[row_i]) == 3:
            for column_i in range(n+1):
                if cromwell.matrix[row_i][column_i] == 1:
                    three.append([row_i, column_i])

    for i in range(n):
        for j in range(m - 1):
            if cromwell.matrix[i][j] == 1 and cromwell.matrix[i][j + 1] == 1:
                for k in range(n):
                    if k != i and [k,j] not in three:
                        for p in range(n):
                            if p != i and p != k:
                                if cromwell.matrix[p][j] + cromwell.matrix[p][j + 1] > 1:
                                    return False
                        return True
    return False

mat = Matrix(4, 5, [
    [1,0,1,0,0],  # row 0
    [0,0,0,1,1],  # row 1
    [1,0,1,0,0],  # row 2
    [1,1,0,0,0]   # row 3
])

print(HorizontalOne(mat))

def reidemeister2_delete_simple(cromwell_integer):
    n = len(cromwell_integer)
    m = max(cromwell_integer).bit_length()

    for i in range(n):
        row_i = cromwell_integer[i]
        for j in range(m - 1):
            if (row_i >> (m - 1 - j)) & 1:
                for k in range(j + 1, m):
                    if (row_i >> (m - 1 - k)) & 1:
                        if k - j > 1:
                            # ↓ 아래 방향 (p < i < q)
                            for p in range(n):
                                if (cromwell_integer[p] >> (m - 1 - j)) & 1:
                                    for q in range(p + 1, n):
                                        if (cromwell_integer[q] >> (m - 1 - k)) & 1:
                                            if p < i < q:
                                                return True
                            # ↑ 위 방향 (q < i < p)
                            for p in range(n):
                                if (cromwell_integer[p] >> (m - 1 - j)) & 1:
                                    for q in range(0, p):
                                        if (cromwell_integer[q] >> (m - 1 - k)) & 1:
                                            if q < i < p:
                                                return True
                            # ← 반대 방향: j ↔ k 바꾸기 (p < i < q)
                            for p in range(n):
                                if (cromwell_integer[p] >> (m - 1 - k)) & 1:
                                    for q in range(p + 1, n):
                                        if (cromwell_integer[q] >> (m - 1 - j)) & 1:
                                            if p < i < q:
                                                return True
                            # ↑ 반대 방향 j ↔ k (q < i < p)
                            for p in range(n):
                                if (cromwell_integer[p] >> (m - 1 - k)) & 1:
                                    for q in range(0, p):
                                        if (cromwell_integer[q] >> (m - 1 - j)) & 1:
                                            if q < i < p:
                                                return True
    return False




print(reidemeister2_delete_simple([24, 10, 37, 35, 20]))
print(reidemeister2_delete_simple([20, 35, 37, 10, 24]))

def reidemeister2_delete_complex(cromwell_integer):
    n = len(cromwell_integer)
    for i in range(n):
        for j in range(n-1):
            if cromwell_integer[i][j] == "1":
                for k in range(j+1, n-1):
                    if cromwell_integer[i][k] == "1":
                        if k-j > 1:
                            for p1 in range(i, n-4):
                                for p2 in range(j, n-1):
                                    if cromwell_integer[p1][p2] == 1:
                                        for q2 in range(p2, n-1):
                                            if cromwell_integer[p1][q2] == 1:
                                                