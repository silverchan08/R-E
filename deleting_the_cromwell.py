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

def cromwell_binary_list(cromwell):
    n = len(cromwell)
    binary_list = []
    for i in range(n):
        a=0
        for j in range(n+1):
            a += cromwell[i][j] * (2**(n+1-j))
        binary_list.append(a)
    return binary_list

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


def reidemeister2_delete(cromwell_integer):
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

print(cromwell_binary_list([[1,0,1,0,0,1,0],[0,0,0,1,0,0,1],[0,0,1,1,0,0,0],[1,1,0,0,1,0,0],[0,0,0,0,1,0,1],[0,1,0,0,0,1,0]]))
print(cromwell_binary_list([[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,0,1],[0,1,0,1,1,0],[1,0,0,0,1,1]]))


def get_bit(num, bit_pos):
    """
    정수 num의 bit_pos 위치의 비트 값을 반환합니다.
    (0-indexed, 가장 오른쪽 비트가 0)
    예: num = 5 (101), bit_pos = 0 -> 1, bit_pos = 1 -> 0, bit_pos = 2 -> 1
    """
    return (num >> bit_pos) & 1

def can_apply_reidemeister3_bitmask(input_rows):
    """
    비트마스킹된 정수 리스트로 주어진 격자 다이어그램에서
    라이데마이스터 3 변환이 가능한 특정 패턴을 탐지합니다.

    Args:
        input_rows (list of int): 각 정수가 격자 다이어그램의 한 행을 나타내며,
                                  정수의 각 비트가 해당 칸에 꼭짓점(1)이 있는지 없는지(0)를 의미합니다.

    Returns:
        bool: 라이데마이스터 3 변환이 가능한 패턴을 찾으면 True, 그렇지 않으면 False.
    """
    n_rows = len(input_rows)

    # 행렬의 최대 열 크기를 추정합니다.
    # 가장 큰 정수의 비트 길이를 기준으로 합니다.
    max_val = 0
    if input_rows:
        max_val = max(input_rows)
    
    # max_val이 0일 경우, 최소 3열 (예: 2^2 = 4)은 있어야 3x3 패턴을 찾을 수 있습니다.
    n_cols = max(3, max_val.bit_length()) 

    # 라이데마이스터 3 변환은 최소 3x3 영역에서 세 개의 꼬임이 얽힐 때 발생합니다.
    # 우리는 이 3x3 영역을 순회하며 특정 패턴을 찾을 것입니다.
    for r_start in range(n_rows - 2):
        for c_start in range(n_cols - 2):
            # 현재 3x3 부분 격자 추출 (비트마스킹된 값에서 비트 추출)
            # sub_grid[row_offset][col_offset]
            sub_grid = [
                [get_bit(input_rows[r_start], c_start), get_bit(input_rows[r_start], c_start + 1), get_bit(input_rows[r_start], c_start + 2)],
                [get_bit(input_rows[r_start + 1], c_start), get_bit(input_rows[r_start + 1], c_start + 1), get_bit(input_rows[r_start + 1], c_start + 2)],
                [get_bit(input_rows[r_start + 2], c_start), get_bit(input_rows[r_start + 2], c_start + 1), get_bit(input_rows[r_start + 2], c_start + 2)]
            ]

            # --- 라이데마이스터 3 변환 패턴 (이전과 동일) ---
            # 1 0 1
            # 0 1 0
            # 1 0 1
            if (sub_grid[0][0] == 1 and sub_grid[0][1] == 0 and sub_grid[0][2] == 1 and
                sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                sub_grid[2][0] == 1 and sub_grid[2][1] == 0 and sub_grid[2][2] == 1):
                print(f"라이데마이스터 3 변환 패턴 1 발견 (시작점: ({r_start},{c_start}))")
                return True

            # 0 1 0
            # 1 0 1
            # 0 1 0
            if (sub_grid[0][0] == 0 and sub_grid[0][1] == 1 and sub_grid[0][2] == 0 and
                sub_grid[1][0] == 1 and sub_grid[1][1] == 0 and sub_grid[1][2] == 1 and
                sub_grid[2][0] == 0 and sub_grid[2][1] == 1 and sub_grid[2][2] == 0):
                print(f"라이데마이스터 3 변환 패턴 2 발견 (시작점: ({r_start},{c_start}))")
                return True
            
            # 1 0 0
            # 0 1 0
            # 0 0 1
            if (sub_grid[0][0] == 1 and sub_grid[0][1] == 0 and sub_grid[0][2] == 0 and
                sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                sub_grid[2][0] == 0 and sub_grid[2][1] == 0 and sub_grid[2][2] == 1):
                print(f"라이데마이스터 3 변환 패턴 3 발견 (시작점: ({r_start},{c_start}))")
                return True
            
            # 0 0 1
            # 0 1 0
            # 1 0 0
            if (sub_grid[0][0] == 0 and sub_grid[0][1] == 0 and sub_grid[0][2] == 1 and
                sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                sub_grid[2][0] == 1 and sub_grid[2][1] == 0 and sub_grid[2][2] == 0):
                print(f"라이데마이스터 3 변환 패턴 4 발견 (시작점: ({r_start},{c_start}))")
                return True

    return False


### 테스트 실행

# 첫 번째 행렬 (False가 나와야 함)
# 각 행을 이진수로 변환:
# [1,0,1,0,0,1,0] -> 64 + 16 + 2 = 82
# [0,0,0,1,0,0,1] -> 8 + 1 = 9
# [0,0,1,1,0,0,0] -> 16 + 8 = 24
# [1,1,0,0,1,0,0] -> 64 + 32 + 4 = 100
# [0,0,0,0,1,0,1] -> 4 + 1 = 5
# [0,1,0,0,0,1,0] -> 32 + 2 = 34
bitmask_matrix1 = [82, 9, 24, 100, 5, 34]
print("--- 첫 번째 행렬 (비트마스킹) ---")
print(can_apply_reidemeister3_bitmask(bitmask_matrix1))

print("\n" + "="*30 + "\n")

# 두 번째 행렬 (True가 나와야 함)
# 각 행을 이진수로 변환:
# [1,0,1,0,0,0] -> 32 + 8 = 40
# [0,1,0,1,0,0] -> 16 + 4 = 20
# [0,0,1,0,0,1] -> 8 + 1 = 9
# [0,1,0,1,1,0] -> 16 + 4 + 2 = 22
# [1,0,0,0,1,1] -> 32 + 2 + 1 = 35
bitmask_matrix2 = [40, 20, 9, 22, 35]
print("--- 두 번째 행렬 (비트마스킹) ---")
print(can_apply_reidemeister3_bitmask(bitmask_matrix2))

print(can_apply_reidemeister3_bitmask([164, 18, 48, 200, 10, 68]))
print(can_apply_reidemeister3_bitmask([80, 40, 18, 44, 70]))