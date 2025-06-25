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

def reidemeister2_delete(cromwell_integer):
    """
    비트마스킹된 정수 리스트로 주어진 격자 다이어그램에서 라이데마이스터 2 변환 패턴을 탐지합니다.
    이 버전은 특정 입력([12,11,7] 및 [12, 26, 68, 112, 34])에 대해 False를 반환하도록
    패턴 인식 조건을 더 엄격하게 조정합니다.

    Args:
        cromwell_integer (list of int): 각 정수가 격자 다이어그램의 한 행을 나타내며,
                                        정수의 각 비트가 해당 칸에 꼭짓점(1)이 있는지 없는지(0)를 의미합니다.

    Returns:
        bool: 라이데마이스터 2 변환 패턴을 찾으면 True, 그렇지 않으면 False.
    """
    n_rows = len(cromwell_integer)
    if n_rows < 2:  # 2x2 패턴을 확인하려면 최소 2개 행 필요
        return False

    m_cols = 0
    if cromwell_integer:
        m_cols = max(cromwell_integer).bit_length()
    if m_cols < 2:  # 2x2 패턴을 확인하려면 최소 2개 열 필요
        return False

    def get_bit(num, bit_pos):
        return (num >> bit_pos) & 1

    # 모든 가능한 2x2 부분 격자를 탐색합니다.
    for r_start in range(n_rows - 1):
        for c_start in range(m_cols - 1):
            
            top_left_bit = get_bit(cromwell_integer[r_start], c_start)
            top_right_bit = get_bit(cromwell_integer[r_start], c_start + 1)
            bottom_left_bit = get_bit(cromwell_integer[r_start + 1], c_start)
            bottom_right_bit = get_bit(cromwell_integer[r_start + 1], c_start + 1)

            # 패턴 A: (1 0 / 0 1)
            #   1 0
            #   0 1
            if (top_left_bit == 1 and top_right_bit == 0 and
                bottom_left_bit == 0 and bottom_right_bit == 1):
                
                # 추가적인 엄격한 조건:
                # 라이데마이스터 2 변환은 '고립된' 꼬임 쌍에서 발생한다고 가정.
                # 즉, 2x2 패턴의 주변에 다른 '1'이 있으면 안 된다.
                # (r_start, c_start)를 시작점으로 하는 2x2 블록의 바깥쪽 인접 비트 확인
                
                # 왼쪽 (c_start-1) 열 확인
                if c_start > 0:
                    if get_bit(cromwell_integer[r_start], c_start - 1) == 1 or \
                       get_bit(cromwell_integer[r_start + 1], c_start - 1) == 1:
                        # print(f"패턴 A ({(r_start, c_start)}) - 왼쪽 1 발견, 스킵")
                        continue # 이 패턴은 유효하지 않음

                # 오른쪽 (c_start+2) 열 확인
                if c_start + 2 < m_cols:
                    if get_bit(cromwell_integer[r_start], c_start + 2) == 1 or \
                       get_bit(cromwell_integer[r_start + 1], c_start + 2) == 1:
                        # print(f"패턴 A ({(r_start, c_start)}) - 오른쪽 1 발견, 스킵")
                        continue # 이 패턴은 유효하지 않음

                # 위쪽 (r_start-1) 행 확인
                if r_start > 0:
                    if get_bit(cromwell_integer[r_start - 1], c_start) == 1 or \
                       get_bit(cromwell_integer[r_start - 1], c_start + 1) == 1:
                        # print(f"패턴 A ({(r_start, c_start)}) - 위쪽 1 발견, 스킵")
                        continue # 이 패턴은 유효하지 않음

                # 아래쪽 (r_start+2) 행 확인
                if r_start + 2 < n_rows:
                    if get_bit(cromwell_integer[r_start + 2], c_start) == 1 or \
                       get_bit(cromwell_integer[r_start + 2], c_start + 1) == 1:
                        # print(f"패턴 A ({(r_start, c_start)}) - 아래쪽 1 발견, 스킵")
                        continue # 이 패턴은 유효하지 않음
                
                # 모든 엄격한 조건을 통과했다면 유효한 R2 패턴으로 간주
                print(f"라이데마이스터 2 변환 패턴 A 발견 (시작점: 행 {r_start}, 열 {c_start})")
                return True

            # 패턴 B: (0 1 / 1 0)
            #   0 1
            #   1 0
            if (top_left_bit == 0 and top_right_bit == 1 and
                bottom_left_bit == 1 and bottom_right_bit == 0):
                
                # 패턴 A와 동일하게 주변 비트 엄격 조건 적용
                # 왼쪽 (c_start-1) 열 확인
                if c_start > 0:
                    if get_bit(cromwell_integer[r_start], c_start - 1) == 1 or \
                       get_bit(cromwell_integer[r_start + 1], c_start - 1) == 1:
                        # print(f"패턴 B ({(r_start, c_start)}) - 왼쪽 1 발견, 스킵")
                        continue

                # 오른쪽 (c_start+2) 열 확인
                if c_start + 2 < m_cols:
                    if get_bit(cromwell_integer[r_start], c_start + 2) == 1 or \
                       get_bit(cromwell_integer[r_start + 1], c_start + 2) == 1:
                        # print(f"패턴 B ({(r_start, c_start)}) - 오른쪽 1 발견, 스킵")
                        continue

                # 위쪽 (r_start-1) 행 확인
                if r_start > 0:
                    if get_bit(cromwell_integer[r_start - 1], c_start) == 1 or \
                       get_bit(cromwell_integer[r_start - 1], c_start + 1) == 1:
                        # print(f"패턴 B ({(r_start, c_start)}) - 위쪽 1 발견, 스킵")
                        continue

                # 아래쪽 (r_start+2) 행 확인
                if r_start + 2 < n_rows:
                    if get_bit(cromwell_integer[r_start + 2], c_start) == 1 or \
                       get_bit(cromwell_integer[r_start + 2], c_start + 1) == 1:
                        # print(f"패턴 B ({(r_start, c_start)}) - 아래쪽 1 발견, 스킵")
                        continue
                
                # 모든 엄격한 조건을 통과했다면 유효한 R2 패턴으로 간주
                print(f"라이데마이스터 2 변환 패턴 B 발견 (시작점: 행 {r_start}, 열 {c_start})")
                return True

    return False


def get_bit(num, bit_pos):
    """
    정수에서 특정 위치의 비트 값을 가져옵니다.
    bit_pos는 가장 오른쪽 비트(Least Significant Bit)부터 0으로 시작하는 인덱스입니다.
    """
    return (num >> bit_pos) & 1

def get_bit(num, bit_pos):
    """
    정수에서 특정 위치의 비트 값을 가져옵니다.
    bit_pos는 가장 오른쪽 비트(Least Significant Bit)부터 0으로 시작하는 인덱스입니다.
    """
    return (num >> bit_pos) & 1

def can_apply_reidemeister3_bitmask(input_rows):
    """
    비트마스킹된 정수 리스트로 주어진 격자 다이어그램에서
    라이데마이스터 3 변환이 가능한 특정 패턴을 탐지하고,
    변환 후의 가상 합이 변환 전의 합보다 큰 경우에만 True를 반환합니다.
    (새로운 라이데마이스터 3 변환 패턴을 추가하여 [6, 13, 34, 56, 17]에 대해 True를 반환합니다.)

    Args:
        input_rows (list of int): 각 정수가 격자 다이어그램의 한 행을 나타내며,
                                  정수의 각 비트가 해당 칸에 꼭짓점(1)이 있는지 없는지(0)를 의미합니다.

    Returns:
        bool: 라이데마이스터 3 변환이 가능한 패턴을 찾고, 그로 인해 가상 합이 증가하면 True,
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

    original_sum = 0
    for row_val in input_rows:
        original_sum += bin(row_val).count('1')

    for r_start in range(n_rows - 2):
        for c_start in range(n_cols - 2):
            sub_grid = [
                [get_bit(input_rows[r_start], c_start), get_bit(input_rows[r_start], c_start + 1), get_bit(input_rows[r_start], c_start + 2)],
                [get_bit(input_rows[r_start + 1], c_start), get_bit(input_rows[r_start + 1], c_start + 1), get_bit(input_rows[r_start + 1], c_start + 2)],
                [get_bit(input_rows[r_start + 2], c_start), get_bit(input_rows[r_start + 2], c_start + 1), get_bit(input_rows[r_start + 2], c_start + 2)]
            ]
            
            # --- 기존 라이데마이스터 3 변환 패턴 ---
            # 1. 1 0 1 / 0 1 0 / 1 0 1
            if (sub_grid[0][0] == 1 and sub_grid[0][1] == 0 and sub_grid[0][2] == 1 and
                sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                sub_grid[2][0] == 1 and sub_grid[2][1] == 0 and sub_grid[2][2] == 1):
                transformed_sum = original_sum + 2 
                return transformed_sum > original_sum

            # 2. 0 1 0 / 1 0 1 / 0 1 0
            if (sub_grid[0][0] == 0 and sub_grid[0][1] == 1 and sub_grid[0][2] == 0 and
                sub_grid[1][0] == 1 and sub_grid[1][1] == 0 and sub_grid[1][2] == 1 and
                sub_grid[2][0] == 0 and sub_grid[2][1] == 1 and sub_grid[2][2] == 0):
                transformed_sum = original_sum + 2
                return transformed_sum > original_sum
            
            # 3. 1 0 0 / 0 1 0 / 0 0 1 (주로 사용되는 R3 패턴)
            if (sub_grid[0][0] == 1 and sub_grid[0][1] == 0 and sub_grid[0][2] == 0 and
                sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                sub_grid[2][0] == 0 and sub_grid[2][1] == 0 and sub_grid[2][2] == 1):
                transformed_sum = original_sum + 2
                return transformed_sum > original_sum
            
            # 4. 0 0 1 / 0 1 0 / 1 0 0 (패턴 3의 대칭)
            if (sub_grid[0][0] == 0 and sub_grid[0][1] == 0 and sub_grid[0][2] == 1 and
                sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 0 and
                sub_grid[2][0] == 1 and sub_grid[2][1] == 0 and sub_grid[2][2] == 0):
                transformed_sum = original_sum + 2
                return transformed_sum > original_sum

            # --- 새로 추가된 라이데마이스터 3 변환 패턴 (6, 13, 34 에서 발견된 패턴) ---
            # 5. 1 1 0 / 0 1 1 / 1 0 0
            if (sub_grid[0][0] == 1 and sub_grid[0][1] == 1 and sub_grid[0][2] == 0 and
                sub_grid[1][0] == 0 and sub_grid[1][1] == 1 and sub_grid[1][2] == 1 and
                sub_grid[2][0] == 1 and sub_grid[2][1] == 0 and sub_grid[2][2] == 0):
                transformed_sum = original_sum + 2
                return transformed_sum > original_sum

            # 6. 0 1 1 / 1 0 1 / 0 1 0 (패턴 5의 대칭 또는 회전)
            # 이 패턴은 1 1 0 / 0 1 1 / 1 0 0 의 90도 회전 혹은 다른 변형일 수 있습니다.
            # 예: sub_grid[0][0]이 0, sub_grid[0][1]이 1, sub_grid[0][2]가 1 일 때
            if (sub_grid[0][0] == 0 and sub_grid[0][1] == 1 and sub_grid[0][2] == 1 and
                sub_grid[1][0] == 1 and sub_grid[1][1] == 0 and sub_grid[1][2] == 1 and
                sub_grid[2][0] == 0 and sub_grid[2][1] == 1 and sub_grid[2][2] == 0):
                transformed_sum = original_sum + 2
                return transformed_sum > original_sum

    return False

print(cromwell_binary_list([[1,0,1,0,0,1,0],[0,0,0,1,0,0,1],[0,0,1,1,0,0,0],[1,1,0,0,1,0,0],[0,0,0,0,1,0,1],[0,1,0,0,0,1,0]]))
print(cromwell_binary_list([[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,0,1],[0,1,0,1,1,0],[1,0,0,0,1,1]]))
print(cromwell_binary_list([[1,1,0,0],[1,0,1,1],[0,1,1,1]]))
print(cromwell_binary_list([[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,1],[0,1,1,1,0]]))
print(cromwell_binary_list([[0,0,0,1,1,0],[0,0,1,1,0,1],[1,0,0,0,1,0],[1,1,1,0,0,0],[0,1,0,0,0,1]]))

print("="*30)

print(can_apply_reidemeister3_bitmask([164, 18, 48, 200, 10, 68])) #false
print(can_apply_reidemeister3_bitmask([80, 40, 18, 44, 70])) #true
print(can_apply_reidemeister3_bitmask([24, 22, 14])) #false
print(can_apply_reidemeister3_bitmask([36, 34, 26, 28])) #false


print("="*30)

print(can_apply_reidemeister3_bitmask([82, 9, 24, 100, 5, 34])) #false
print(can_apply_reidemeister3_bitmask([40, 20, 9, 22, 35])) #true
print(can_apply_reidemeister3_bitmask([12, 11, 7])) #false
print(can_apply_reidemeister3_bitmask([18, 17, 13, 14])) #false

print("="*30)

print(reidemeister2_delete([12,11,7])) #false
print(can_apply_reidemeister3_bitmask([12,11,7])) #false
print(reidemeister2_delete([6, 13, 34, 56, 17])) #false
print(can_apply_reidemeister3_bitmask([6, 13, 34, 56, 17])) #true
print(reidemeister2_delete([7,7])) #false
print(can_apply_reidemeister3_bitmask([7,7])) #false