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
    binary_list = []
    for row in cromwell: # 각 행(이진수 리스트)을 순회
        decimal_value = 0
        num_bits = len(row) # 현재 행의 비트 수 (여기서는 6)
        for j in range(num_bits):
            decimal_value += row[j] * (2**(num_bits - 1 - j))
        binary_list.append(decimal_value)
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

def get_bit(num, bit_pos):
    """
    정수에서 특정 위치의 비트 값을 가져옵니다.
    bit_pos는 가장 오른쪽 비트(Least Significant Bit)부터 0으로 시작하는 인덱스입니다.
    """
    return (num >> bit_pos) & 1

def set_bit(num, bit_pos, value):
    """
    정수의 특정 위치의 비트를 설정합니다.
    value가 1이면 해당 비트를 1로, 0이면 0으로 만듭니다.
    """
    if value == 1:
        return num | (1 << bit_pos)  # 해당 비트를 1로 설정
    else:
        return num & (~(1 << bit_pos)) # 해당 비트를 0으로 설정

def reidemeister2_delete(cromwell_integer):
    """
    비트마스킹된 정수 리스트에서 순수한 라이데마이스터 2 변환 (생성)을 시뮬레이션합니다.
    변환 후 '첫 번째 원소'의 값이 변환 전보다 커지면 True를 반환합니다.

    Args:
        cromwell_integer (list of int): 각 정수가 격자 다이어그램의 한 행을 나타내며,
                                        정수의 각 비트가 해당 칸에 꼭짓점(1)이 있는지 없는지(0)를 의미합니다.

    Returns:
        bool: 라이데마이스터 2 변환(생성)으로 인해 첫 번째 원소의 값이 증가하면 True,
              그렇지 않으면 False.
    """
    n_rows = len(cromwell_integer)
    if n_rows < 2:  # 2x2 패턴을 확인하려면 최소 2개 행 필요
        return False

    m_cols = 0
    if cromwell_integer:
        # 최대값의 비트 길이를 사용하여 열의 수를 결정합니다. 최소 2개의 열이 필요합니다.
        m_cols = max(2, max(cromwell_integer).bit_length())
    else:
        return False # 빈 리스트인 경우 처리

    # 원본 첫 번째 원소 값을 저장
    original_first_element = cromwell_integer[0]

    # 모든 가능한 2x2 부분 격자를 탐색합니다.
    for r_start in range(n_rows - 1):
        for c_start in range(m_cols - 1):
            
            # 현재 2x2 영역의 비트 상태를 가져옵니다.
            top_left_bit = get_bit(cromwell_integer[r_start], c_start)
            top_right_bit = get_bit(cromwell_integer[r_start], c_start + 1)
            bottom_left_bit = get_bit(cromwell_integer[r_start + 1], c_start)
            bottom_right_bit = get_bit(cromwell_integer[r_start + 1], c_start + 1)

            # --- 라이데마이스터 2 변환 (생성) 시뮬레이션 ---
            # 비어있는 2x2 영역 (0 0 / 0 0)을 찾습니다.
            if (top_left_bit == 0 and top_right_bit == 0 and
                bottom_left_bit == 0 and bottom_right_bit == 0):
                

                # 1. 패턴 A (1 0 / 0 1) 생성 시도
                # 원본을 변경하지 않기 위해 현재 상태의 cromwell_integer 복사
                temp_grid_rows_a = list(cromwell_integer) 
                
                # 해당 2x2 영역에 패턴 A를 '생성'합니다.
                temp_grid_rows_a[r_start] = set_bit(temp_grid_rows_a[r_start], c_start, 1)     # (r, c) = 1
                temp_grid_rows_a[r_start] = set_bit(temp_grid_rows_a[r_start], c_start + 1, 0) # (r, c+1) = 0
                temp_grid_rows_a[r_start + 1] = set_bit(temp_grid_rows_a[r_start + 1], c_start, 0) # (r+1, c) = 0
                temp_grid_rows_a[r_start + 1] = set_bit(temp_grid_rows_a[r_start + 1], c_start + 1, 1) # (r+1, c+1) = 1

                # 변환 후의 첫 번째 원소 값 확인
                # 변환이 첫 번째 행(인덱스 0)에 영향을 주는 경우에만 값 증가 여부 확인.
                # r_start가 0이면 temp_grid_rows_a[0]의 값이 바뀔 수 있습니다.
                if r_start == 0:
                    i = 0
                    while i <= n_rows:
                        if temp_grid_rows_a[i] > original_first_element:
                            return True
                        elif temp_grid_rows_a[i] == original_first_element:
                            i += 1
                        
                
                # 2. 패턴 B (0 1 / 1 0) 생성 시도
                # 다시 원본을 변경하지 않기 위해 현재 상태의 cromwell_integer 복사
                temp_grid_rows_b = list(cromwell_integer)
                
                # 해당 2x2 영역에 패턴 B를 '생성'합니다.
                temp_grid_rows_b[r_start] = set_bit(temp_grid_rows_b[r_start], c_start, 0)     # (r, c) = 0
                temp_grid_rows_b[r_start] = set_bit(temp_grid_rows_b[r_start], c_start + 1, 1) # (r, c+1) = 1
                temp_grid_rows_b[r_start + 1] = set_bit(temp_grid_rows_b[r_start + 1], c_start, 1) # (r+1, c) = 1
                temp_grid_rows_b[r_start + 1] = set_bit(temp_grid_rows_b[r_start + 1], c_start + 1, 0) # (r+1, c+1) = 0

                # 변환 후의 첫 번째 원소 값 확인
                if r_start == 0:
                    i = 0
                    while i <= n_rows:
                        if temp_grid_rows_b[i] > original_first_element:
                            return True
                        elif temp_grid_rows_b[i] == original_first_element:
                            i += 1

                
    # 어떤 패턴도 발견되지 않거나, 생성 후 첫 번째 원소가 증가하지 않으면 False 반환
    return False


def get_bit(num, bit_pos):
    """
    정수에서 특정 위치의 비트 값을 가져옵니다.
    bit_pos는 가장 오른쪽 비트(Least Significant Bit)부터 0으로 시작하는 인덱스입니다.
    """
    return (num >> bit_pos) & 1

def can_apply_reidemeister3_bitmask(input_rows):
    """
    비트마스킹된 정수 리스트로 주어진 격자 다이어그램에서
    라이데마이스터 3 변환을 시뮬레이션합니다.
    변환 후 리스트의 첫 번째 원소(input_rows[0])의 값이 변환 전보다 커지면 True를 반환합니다.
    만약 첫 번째 원소의 값이 같다면, 두 번째 원소(input_rows[1])의 값이
    변환 전보다 커지는지 확인하여 크면 True를 반환합니다.

    Args:
        input_rows (list of int): 각 정수가 격자 다이어그램의 한 행을 나타내며,
                                  정수의 각 비트가 해당 칸에 꼭짓점(1)이 있는지 없는지(0)를 의미합니다.

    Returns:
        bool: 라이데마이스터 3 변환이 가능한 패턴을 찾아 적용했을 때,
              첫 번째 원소의 값이 증가하거나, 첫 번째 원소의 값이 같고 두 번째 원소의 값이 증가하면 True,
              그렇지 않으면 False.
    """
    n_rows = len(input_rows)
    if n_rows < 3: # 3x3 패턴을 확인하려면 최소 3개 행 필요
        return False

    max_val = 0
    if input_rows:
        max_val = max(input_rows)
    
    # 3x3 패턴을 찾기 위해 최소 3열이 필요합니다.
    n_cols = max(3, max_val.bit_length()) 

    # 원본 리스트의 첫 번째와 두 번째 원소 값을 저장합니다.

    # 모든 가능한 3x3 부분 격자를 탐색합니다.
    for r_start in range(n_rows - 2):
        for c_start in range(n_cols - 2):
            
            # 현재 3x3 부분 격자의 비트 값을 가져옵니다.
            sub_grid = [
                [get_bit(input_rows[r_start], c_start), get_bit(input_rows[r_start], c_start + 1), get_bit(input_rows[r_start], c_start + 2)],
                [get_bit(input_rows[r_start + 1], c_start), get_bit(input_rows[r_start + 1], c_start + 1), get_bit(input_rows[r_start + 1], c_start + 2)],
                [get_bit(input_rows[r_start + 2], c_start), get_bit(input_rows[r_start + 2], c_start + 1), get_bit(input_rows[r_start + 2], c_start + 2)]
            ]
            
            # 변환 시뮬레이션을 위한 임시 리스트를 미리 준비합니다.
            # 각 패턴마다 새롭게 복사하여 독립적인 변환을 시도합니다.
            temp_rows = list(input_rows) # 초기 복사

            # --- 라이데마이스터 3 변환 패턴 및 시뮬레이션 ---
            # 각 패턴에 대해 변환 로직을 정의하고 결과 확인

            transformed = False # 변환이 실제로 일어났는지 여부

            # 1. 1 0 1 / 0 1 0 / 1 0 1
            if (sub_grid[0][0] == 1 and sub_grid[0][1] == 0 and sub_grid[0][2] == 1 and
                sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                sub_grid[2][0] == 1 and sub_grid[2][1] == 0 and sub_grid[2][2] == 1):
                
                
                # 변환 시뮬레이션: (원본 상태를 패턴 2로 변환하는 것으로 가정)
                # 101 -> 010
                # 010 -> 101
                # 101 -> 010
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start, 0)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 1, 1)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 2, 0)

                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start, 1)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 1, 0)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 2, 1)
                
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start, 0)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 1, 1)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 2, 0)
                transformed = True

            # 2. 0 1 0 / 1 0 1 / 0 1 0
            elif (sub_grid[0][0] == 0 and sub_grid[0][1] == 1 and sub_grid[0][2] == 0 and
                  sub_grid[1][0] == 1 and sub_grid[1][1] == 0 and sub_grid[1][2] == 1 and
                  sub_grid[2][0] == 0 and sub_grid[2][1] == 1 and sub_grid[2][2] == 0):
                
                
                # 변환 시뮬레이션: (원본 상태를 패턴 1로 변환하는 것으로 가정)
                # 010 -> 101
                # 101 -> 010
                # 010 -> 101
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start, 1)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 1, 0)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 2, 1)

                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start, 0)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 1, 1)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 2, 0)
                
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start, 1)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 1, 0)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 2, 1)
                transformed = True

            # 3. 1 0 0 / 0 1 0 / 0 0 1 (주로 사용되는 R3 패턴 - 대각선 이동)
            elif (sub_grid[0][0] == 1 and sub_grid[0][1] == 0 and sub_grid[0][2] == 0 and
                  sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                  sub_grid[2][0] == 0 and sub_grid[2][1] == 0 and sub_grid[2][2] == 1):
                
                
                # 변환 시뮬레이션: (원본 상태를 패턴 4로 변환하는 것으로 가정)
                # 100 -> 001
                # 010 -> 010
                # 001 -> 100
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start, 0)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 1, 0)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 2, 1)

                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start, 0)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 1, 1)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 2, 0)
                
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start, 1)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 1, 0)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 2, 0)
                transformed = True

            # 4. 0 0 1 / 0 1 0 / 1 0 0 (패턴 3의 대칭)
            elif (sub_grid[0][0] == 0 and sub_grid[0][1] == 0 and sub_grid[0][2] == 1 and
                  sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                  sub_grid[2][0] == 1 and sub_grid[2][1] == 0 and sub_grid[2][2] == 0):
                
                
                # 변환 시뮬레이션: (원본 상태를 패턴 3으로 변환하는 것으로 가정)
                # 001 -> 100
                # 010 -> 010
                # 100 -> 001
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start, 1)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 1, 0)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 2, 0)

                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start, 0)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 1, 1)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 2, 0)
                
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start, 0)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 1, 0)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 2, 1)
                transformed = True

            # 5. 1 1 0 / 0 1 1 / 1 0 0 (새롭게 추가된 패턴)
            elif (sub_grid[0][0] == 1 and sub_grid[0][1] == 1 and sub_grid[0][2] == 0 and
                  sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 1 and
                  sub_grid[2][0] == 1 and sub_grid[2][1] == 0 and sub_grid[2][2] == 0):
                
                
                # 이 패턴에 대한 R3 변환 예시를 가정합니다. (패턴 6으로 변환)
                # 110 -> 011
                # 011 -> 101
                # 100 -> 010
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start, 0)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 1, 1)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 2, 1)

                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start, 1)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 1, 0)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 2, 1)
                
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start, 0)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 1, 1)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 2, 0)
                transformed = True

            # 6. 0 1 1 / 1 0 1 / 0 1 0 (패턴 5의 대칭 또는 회전)
            elif (sub_grid[0][0] == 0 and sub_grid[0][1] == 1 and sub_grid[0][2] == 1 and
                  sub_grid[1][0] == 1 and sub_grid[1][1] == 0 and sub_grid[1][2] == 1 and
                  sub_grid[2][0] == 0 and sub_grid[2][1] == 1 and sub_grid[2][2] == 0):
                
                
                # 이 패턴에 대한 R3 변환 예시를 가정합니다. (패턴 5로 변환)
                # 011 -> 110
                # 101 -> 011
                # 010 -> 100
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start, 1)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 1, 1)
                temp_rows[r_start] = set_bit(temp_rows[r_start], c_start + 2, 0)

                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start, 0)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 1, 1)
                temp_rows[r_start + 1] = set_bit(temp_rows[r_start + 1], c_start + 2, 1)
                
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start, 1)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 1, 0)
                temp_rows[r_start + 2] = set_bit(temp_rows[r_start + 2], c_start + 2, 0)
                transformed = True

            # 변환이 일어났다면 (어떤 R3 패턴이든 발견되어 시뮬레이션되었다면) 결과 확인
            if transformed:
                i = 0
                while i < n_rows-1:
                    if temp_rows[i] > input_rows[i]:
                        return True
                    elif temp_rows[i] < input_rows[i]:
                        break
                    elif temp_rows[i] == input_rows[i]:
                        i += 1

                
                # 다음 패턴 탐색을 위해 temp_rows를 다시 원본으로 초기화할 필요 없음.
                # 다음 루프에서 sub_grid를 다시 input_rows에서 가져오므로 문제 없음.
                # 그러나 이 함수는 "어떤 한 번의 변환으로라도" 증가하면 True를 반환하므로,
                # 한 패턴을 적용 후 True가 아니면 바로 다음 패턴으로 넘어가는 것이 효율적입니다.
                # (continue 대신 if-else if 구조 사용)
                
    # 모든 패턴을 탐색했음에도 조건을 만족하는 변환을 찾지 못하면 False 반환
    return False


print(cromwell_binary_list([[1,0,1,0,0,1,0],[0,0,0,1,0,0,1],[0,0,1,1,0,0,0],[1,1,0,0,1,0,0],[0,0,0,0,1,0,1],[0,1,0,0,0,1,0]]))
print(cromwell_binary_list([[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,0,1],[0,1,0,1,1,0],[1,0,0,0,1,1]]))
print(cromwell_binary_list([[1,1,0,0],[1,0,1,1],[0,1,1,1]]))
print(cromwell_binary_list([[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,1],[0,1,1,1,0]]))
print(cromwell_binary_list([[0,0,0,1,1,0],[0,0,1,1,0,1],[1,0,0,0,1,0],[1,1,1,0,0,0],[0,1,0,0,0,1]]))

print("="*30)

print(can_apply_reidemeister3_bitmask([82, 9, 24, 100, 5, 34])) #false
print(can_apply_reidemeister3_bitmask([40, 20, 9, 22, 35])) #true
print(can_apply_reidemeister3_bitmask([12, 11, 7])) #false
print(can_apply_reidemeister3_bitmask([18, 17, 13, 14])) #false

print("="*30)

print(reidemeister2_delete([12,11,7])) #false
print(can_apply_reidemeister3_bitmask([12,11,7])) #false
print(reidemeister2_delete([6, 13, 34, 56, 17])) #true
print(can_apply_reidemeister3_bitmask([6, 13, 34, 56, 17])) #true
print(reidemeister2_delete([7,7])) #false
print(can_apply_reidemeister3_bitmask([7,7])) #false

print("="*30)

print(reidemeister2_delete([14, 17, 9, 22])) #false
print(can_apply_reidemeister3_bitmask([14, 17, 9, 22])) #false