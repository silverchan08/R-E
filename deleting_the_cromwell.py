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

def cromwell_binary_list(cromwell):
    binary_list = []
    for row in cromwell: # 각 행(이진수 리스트)을 순회
        decimal_value = 0
        num_bits = len(row) # 현재 행의 비트 수 (여기서는 6)
        for j in range(num_bits):
            decimal_value += row[j] * (2**(num_bits - 1 - j))
        binary_list.append(decimal_value)
    return binary_list

def symmetry_delete(input_rows):
    """
    비트마스킹된 정수 리스트(행)로 주어진 격자 다이어그램에 대해
    상하 반전, 좌우 반전, 상하좌우 반전 대칭 변환을 수행합니다.

    각 대칭 변환된 결과의 첫 번째 원소(인덱스 0)가
    원본 input_rows의 첫 번째 원소보다 큰 값이 하나라도 있으면 False를 반환합니다.
    모든 대칭 변환된 결과의 첫 번째 원소가 원본 input_rows의 첫 번째 원소보다 작거나 같으면 True를 반환합니다.

    Args:
        input_rows (list of int): 각 정수가 격자 다이어그램의 한 행을 나타내며,
                                  정수의 각 비트가 해당 칸에 꼭짓점(1)이 있는지 없는지(0)를 의미합니다.

    Returns:
        bool: 모든 대칭 변환 결과의 첫 번째 원소가 원본 input_rows[0]보다 작거나 같으면 True,
              하나라도 큰 값이 있으면 False.
    """
    n_rows = len(input_rows)  # 행의 개수 (정수 리스트의 길이)
     


    # 열의 개수를 결정합니다. 모든 정수의 최대 비트 길이를 기준으로 합니다.
    # 만약 모든 정수가 0이라면 최소 1열로 간주합니다.
    n_cols = 0
    if input_rows: # input_rows가 비어있지 않은 경우
        # max_val이 0인 경우 bit_length()는 0을 반환하므로, 최소 1로 보장합니다.
        n_cols = max(num.bit_length() for num in input_rows) if any(input_rows) else 1
    

    # --- 1. 상하 반전 (Vertical Flip) ---
    flipped_vertical = [0] * n_rows
    for i in range(n_rows):
        flipped_vertical[i] = input_rows[n_rows - 1 - i]
    
    while i < n_rows-1:
        if flipped_vertical[i] > input_rows[i]:
            return False
        elif flipped_vertical[i] < input_rows:
            break
        elif flipped_vertical[i] == input_rows[i]:
            i += 1


    # --- 2. 좌우 반전 (Horizontal Flip) ---
    flipped_horizontal = [0] * n_rows
    for i in range(n_rows):
        new_row_val = 0
        for j in range(n_cols):
            # 원본 input_rows[i]의 비트를 역순으로 가져와 새로운 행에 설정합니다.
            bit_val = get_bit(input_rows[i], n_cols - 1 - j)
            new_row_val = set_bit(new_row_val, j, bit_val)
        flipped_horizontal[i] = new_row_val
    
    while i < n_rows-1:
        if flipped_horizontal[i] > input_rows[i]:
            return False
        elif flipped_horizontal[i] < input_rows:
            break
        elif flipped_horizontal[i] == input_rows[i]:
            i += 1

    # --- 3. 상하좌우 반전 (Diagonal Flip - typically 180 degree rotation, or combined vertical & horizontal) ---
    # 상하 반전된 상태에서 다시 좌우 반전을 수행합니다.
    flipped_diagonal = [0] * n_rows
    for i in range(n_rows):
        # 상하 반전될 원본 행의 비트를 가져옵니다.
        original_row_for_diagonal = input_rows[n_rows - 1 - i] 
        new_row_val = 0
        for j in range(n_cols):
            # 가져온 비트를 좌우 반전하여 새로운 행에 설정합니다.
            bit_val = get_bit(original_row_for_diagonal, n_cols - 1 - j)
            new_row_val = set_bit(new_row_val, j, bit_val)
        flipped_diagonal[i] = new_row_val

    while i < n_rows-1:
        if flipped_diagonal[i] > input_rows[i]:
            return False
        elif flipped_diagonal[i] < input_rows:
            break
        elif flipped_diagonal[i] == input_rows[i]:
            i += 1

    return True


def VerticalOne(cromwell):
    """
    비트마스킹된 정수 리스트(행)로 주어진 격자 다이어그램에서
    연속된 수직 1 패턴을 확인합니다.

    다음 조건을 만족하는 경우 True를 반환합니다:
    1. 인접한 두 행(i와 i+1)의 같은 열(j)에 모두 비트 1이 있습니다.
    2. 해당 두 행(i와 i+1)이 각각 '세 개의 비트 1을 포함하는 행'이 아닙니다.
    이 외의 경우에는 False를 반환합니다.

    Args:
        cromwell (list of int): 각 정수가 격자 다이어그램의 한 행을 나타내며,
                                정수의 각 비트가 해당 칸에 꼭짓점(1)이 있는지 없는지(0)를 의미합니다.

    Returns:
        bool: 조건을 만족하는 수직 1 패턴이 발견되면 True, 그렇지 않으면 False.
    """
    n_rows = len(cromwell) # 행의 개수

    # 열의 개수를 결정합니다. 모든 정수의 최대 비트 길이를 기준으로 합니다.
    # 만약 모든 정수가 0이라면 최소 1열로 간주합니다.
    n_cols = 0
    if cromwell:
        n_cols = max(num.bit_length() for num in cromwell) if any(cromwell) else 1
    

    # 1. 세 개의 비트 1을 포함하는 행들의 인덱스를 찾습니다.
    # (원본 코드의 'sum(cromwell.matrix[row_i]) == 3'에 해당)
    three_ones_rows_indices = []
    for row_i in range(n_rows):
        # 각 행(정수)의 비트를 세는 효율적인 방법: bin(num).count('1')
        if bin(cromwell[row_i]).count('1') == 3:
            three_ones_rows_indices.append(row_i)
    

    # 2. 연속된 수직 1 패턴을 확인합니다.
    for i in range(n_rows - 1): # i는 0부터 n_rows - 2까지 (i+1에 접근하기 위함)
        for j in range(n_cols): # j는 0부터 n_cols - 1까지 (각 열 확인)
            # 현재 행(i)과 다음 행(i+1)의 j번째 비트가 모두 1인지 확인
            is_vertical_one = (get_bit(cromwell[i], j) == 1 and
                               get_bit(cromwell[i + 1], j) == 1)

            # 해당 두 행이 '세 개의 1을 포함하는 행'이 아닌지 확인
            rows_not_in_three_ones = (i not in three_ones_rows_indices and
                                      (i + 1) not in three_ones_rows_indices)
            
            if is_vertical_one and rows_not_in_three_ones:
                return True
    
    return False



def HorizontalOne(cromwell):
    """
    비트마스킹된 정수 리스트(행)로 주어진 격자 다이어그램에서
    "위아래 가로줄이 똑같으면 지우는" 로직을 시뮬레이션한 후 특정 조건을 확인합니다.

    로직:
    1. '세 개의 1'을 가진 행의 비트 위치를 `three_ones_coords`에 저장합니다.
    2. 모든 행 `i`에 대해 `cromwell[i]`와 `cromwell[i+1]`이 같은지 확인합니다.
    3. 만약 같다면:
        a. **임시 다이어그램에서 이 두 행을 제거한 것을 시뮬레이션합니다.**
        b. 제거된 행들이 `three_ones_coords`와 관련된 조건을 만족하는지 확인합니다.
        c. 남은 임시 다이어그램의 행들에 대해 원본 `p` 루프와 유사한 조건 (`j`열과 `j+1`열에 모두 1이 있는지)을 확인합니다.
           이 조건에 위배되면 `False`를 반환합니다.
        d. 모든 조건을 통과하면 `True`를 반환합니다.

    Args:
        cromwell (list of int): 각 정수가 격자 다이어그램의 한 행을 나타내며,
                                정수의 각 비트가 해당 칸에 꼭짓점(1)이 있는지 없는지(0)를 의미합니다.

    Returns:
        bool: 조건을 만족하는 패턴이 발견되면 True, 그렇지 않으면 False.
    """
    n_rows = len(cromwell)

    if n_rows < 2: # 최소 2개 행이 있어야 위아래 비교 및 제거가 가능합니다.
        return False

    n_cols = 0
    if cromwell:
        n_cols = max((num.bit_length() for num in cromwell), default=1)
    

    # '세 개의 1'을 가진 행에서 각 1의 위치를 저장합니다.
    three_ones_coords = [] # [[row_idx, col_idx], ...]
    for row_i in range(n_rows):
        if bin(cromwell[row_i]).count('1') == 3:
            for col_j in range(n_cols):
                if get_bit(cromwell[row_i], col_j) == 1:
                    three_ones_coords.append([row_i, col_j])
    

    # 모든 행 i에 대해 cromwell[i]와 cromwell[i+1]이 같은지 확인합니다.
    for i in range(n_rows - 1): # i는 0부터 n_rows - 2까지
        if cromwell[i] == cromwell[i + 1]:
            
            # --- 이제 이 두 행을 "지운" 상태를 시뮬레이션하고 조건을 확인합니다. ---
            
            # 임시 다이어그램 생성 (제거된 행 제외)
            temp_cromwell = []
            for idx, row_val in enumerate(cromwell):
                if idx != i and idx != (i + 1): # 동일한 두 행을 제외하고 복사
                    temp_cromwell.append(row_val)
            

            # 원본 k 루프 해석: "3개짜리 제외" 조건
            # 원본: for k in range(n): if k != i and [k,j] not in three:
            # 여기서는 제거된 행 i, i+1에 해당하는 `j` 열이 `three_ones_coords`에 없어야 한다는 의미로 해석
            # `three_ones_coords`는 원본 다이어그램 기준으로 만들어졌으므로,
            # 제거되는 행의 비트 위치가 `three_ones_coords`에 포함되는지 확인합니다.
            
            # 이 'k' 루프는 원본 코드에서 `j`에 대한 반복문 안에 있었습니다.
            # 즉, 특정 '수평 11' 패턴이 발견되면, 그 'j' 열에 대해 `k` 조건이 검사되었습니다.
            # 지금은 '동일한 가로줄'을 찾았으므로, 이 'k' 조건이 'j'와 어떻게 연결될지 다시 생각해 봐야 합니다.
            # 가장 합리적인 해석은 "제거된 행의 특정 열(`j`)이 `three_ones_coords`에 없어야 한다"는 것.
            
            # 원본 `HorizontalOne`의 `for j in range(m-1)` 루프가 전체를 감싸고 있었고,
            # `if cromwell.matrix[i][j] == 1 and cromwell.matrix[i][j + 1] == 1:` 이 조건 다음에 `k`와 `p`가 나왔습니다.
            # 이는 `j`와 `j+1`이라는 특정 열 쌍에 대해 검사한다는 의미입니다.
            
            # 따라서, 동일한 행 i와 i+1을 발견했다면, 이 두 행의 모든 열에 대해 원본 k, p 조건을 검사해야 합니다.
            # 그러나 원본 코드는 `cromwell.matrix[i][j] == 1 and cromwell.matrix[i][j + 1] == 1`을
            # 만족하는 `j`를 찾은 후에야 `k`와 `p` 루프에 들어갔습니다.
            # 이것은 "지우는 것"이 아니라, "특정 패턴이 있는 행을 중심으로 검사"하는 것에 가깝습니다.

            # 사용자님의 "가로줄을 지우고 세로줄을 합치는" 이 새로운 설명과
            # 원본 코드의 `cromwell.matrix[i][j] == 1 and cromwell.matrix[i][j + 1] == 1` 조건을 동시에 만족시키려면,
            # 두 가지 방식이 있을 수 있습니다:
            # 1. '11' 패턴을 가진 행(i)을 찾고, 그 행 i와 동일한 비트 패턴을 가진 i+1이 있다면,
            #    이 i, i+1 행을 제거한 후 남은 다이어그램을 대상으로 `three_ones_coords`와 `p` 조건을 검사.
            # 2. '11' 패턴을 가진 행(i)을 찾고, 그 행 i가 다른 어떤 행 i+1과 동일하다면,
            #    그 i, i+1 행이 '세 개의 1을 포함하는 위치'에 없으면서,
            #    다른 모든 p 행에서 j, j+1 열이 동시에 1인 경우가 없어야 한다.

            # 사용자님의 최신 "위아래 가로줄이 똑같으면 지우는 거야"를 우선하여 2번 방식으로 구현합니다.
            # 즉, 특정 열 `j`, `j+1`에 **동시에 1이 있는 경우 (연속된 11 패턴)** 에 대해 이 행(`i`)과 `i+1`이 동일한 경우를 찾고,
            # 그 경우에 `k`, `p` 조건을 적용합니다.

            # 2025-06-25 20:12 KST 기준:
            # "위아래 가로줄이 똑같으면 지우는 거야"라는 말은
            # `if cromwell[i] == cromwell[i+1]`이 True일 때,
            # 원본 `HorizontalOne` 함수의 나머지 복잡한 조건들을 확인하는 것으로 해석합니다.
            # 그리고 `cromwell.matrix[i][j] == 1 and cromwell.matrix[i][j + 1] == 1` 조건은
            # '지울 가로줄'의 특성으로 사용하고, 이 지워지는 가로줄이 특정 열 `j`에 대해 11 패턴을 가져야 하는 것으로 봅니다.

            for j in range(n_cols - 1): # 동일한 두 행 i와 i+1이 발견된 경우, 그 행들의 모든 열 쌍(j, j+1)에 대해
                # 원본 코드의 if cromwell.matrix[i][j] == 1 and cromwell.matrix[i][j + 1] == 1: 조건.
                # 이는 "지울 가로줄"이 특정 위치(j, j+1)에서 11 패턴을 가지는지 확인하는 것.
                if get_bit(cromwell[i], j) == 1 and get_bit(cromwell[i], j + 1) == 1:

                    # 원본 'k' 루프 조건:
                    # 'k != i' 이고, '[k,j]' (행 k의 j열에 있는 비트)가 three_ones_coords에 없어야 함.
                    # 여기서는 'k'는 제거된 행 `i`와 `i+1`을 제외한 다른 행들을 의미합니다.
                    # 원본 코드에서는 `k`가 `j`열에 1이 있는지를 `cromwell.matrix[k][j]`로 직접 확인했지만,
                    # 여기서는 `[k,j]`가 `three_ones_coords`에 있는지 여부만 확인합니다.
                    
                    k_condition_satisfied = True
                    for k_row_idx in range(n_rows):
                        if k_row_idx != i and k_row_idx != (i + 1): # 제거된 두 행 제외
                            if [k_row_idx, j] in three_ones_coords:
                                k_condition_satisfied = False
                                break
                    
                    if not k_condition_satisfied:
                        continue # 이 `j`에 대한 검사를 실패했으므로 다음 `j`로 넘어감

                    # 원본 'p' 루프 조건:
                    # 'p != i' 이고 'p != k' 일 때,
                    # `cromwell.matrix[p][j] + cromwell.matrix[p][j + 1] > 1` 이면 `return False`.
                    # 이는 "지워지지 않은 다른 행 `p`에서 `j`열과 `j+1`열에 모두 1이 있으면 안 된다"는 의미로 해석됩니다.

                    p_condition_satisfied = True
                    for p_row_idx in range(n_rows):
                        if p_row_idx != i and p_row_idx != (i + 1): # 제거된 두 행 제외
                            # 'p != k'는 현재 중첩 루프 구조에서 직접적으로 검사하기 어렵지만,
                            # 'k'는 `j`와 연관된 특정 위치를 가리키는 필터링 역할이었으므로,
                            # 'p'는 단순히 제거된 두 행을 제외한 모든 다른 행으로 간주합니다.
                            
                            if get_bit(cromwell[p_row_idx], j) == 1 and get_bit(cromwell[p_row_idx], j + 1) == 1:
                                return False # 즉시 False 반환 (전체 함수 종료)
                    
                    # 모든 k, p 조건을 통과했다면 True 반환
                    return True
    
    return False

def LeftOne(list):
    CP = []
    n = list.bit_length()
    m = list.bit_count()
    for i in range(n+1):
        if (list >> i).bit_count() != m:
            CP.append(i)
            m = (list >> i).bit_count()
    return CP[1]

def R1(list):
    three = []
    n = len(list)
    for row_i in range(n):
        if list[row_i].bit_count() == 3:
            three.append(row_i)
    
    for i in range(n):
        for j in range(i+1,n):
            t = 0
            if (list[i]&list[j]).bit_count() == 1:
                M = max(((list[i]&list[j])^list[i]).bit_length(),((list[i]&list[j])^list[j]).bit_length())
                M2 = min(LeftOne(list[i]),LeftOne(list[j]))
                N = (list[i]&list[j]).bit_length()
                
                if list[i].bit_length() == list[j].bit_length():
                    if (M == ((list[i]&list[j])^list[i]).bit_length() and i in three) or (M == ((list[i]&list[j])^list[j]).bit_length() and j in three):
                        continue
                    else:
                        for k in range(1,j-i):
                            if (list[i+k] >> M-1).bit_count() != (list[i+k] >> N).bit_count():
                                t = 1
                                break
                    if t:
                        break
                    return True
                elif list[i].bit_length()>((list[i]&list[j]).bit_length()) and list[j].bit_length()>((list[i]&list[j]).bit_length()): 
                    if i in three and (list[i] >> (list[i]&list[j]).bit_length()).bit_count() <= 1 or (list[i] >> M2) <= 1:
                        continue
                    elif j in three and (list[j] >> (list[i]&list[j]).bit_length()).bit_count() <= 1 or (list[j] >> M2) <= 1:
                        continue
                    else:
                        for l in range(1,j-i): 
                            if (list[i+l] >> N-1).bit_count() != (list[i+l] >> M2).bit_count():
                                t = 1
                                break
                    if t:
                        break
                    return True
    
    return False

def R2(L):
    n = len(L)
    cromwell = list(list(map(int, ('0'*(n+1-len(bin(L[i])[2:])))+(bin(L[i]))[2:])) for i in range(len(L)))
    print(cromwell)
    for i in range(n):
        if sum(cromwell[i]) == 3:
            continue
        ind = []
        for j in range(n+1):
            if cromwell[i][j]:
                ind.append(j)
        fir, sec = ind[0], ind[1]
        for j in range(n):
            if cromwell[j][fir] == 1 and j != i:
                go1 = j
        for j in range(n):
            if cromwell[j][sec] == 1 and j != i:
                go2 = j
        if go1 < i < go2 or go2< i < go1:
            continue
        elif go1 == go2:
            return True
        elif go1 < i:
            go = max(go1, go2)
            test = 1
            for j in range(go, i):
                if j == go:
                    if sum(cromwell[j][fir:(sec+1)])-1:
                        test = 0
                        break
                else:
                    if sum(cromwell[j][fir:(sec+1)]):
                        test = 0
                        break
            if test:
                return True
        elif i < go1:
            go = min(go1, go2)
            test = 1
            for j in range(i+1,go+1):
                if j == go:
                    if sum(cromwell[j][fir:(sec+1)])-1:
                        test = 0
                        break
                else:
                    if sum(cromwell[j][fir:(sec+1)]):
                        test = 0
                        break
            if test:
                return True
    return False



def R3(input_rows):
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
                if temp_rows > input_rows:
                    return True
    return False


print(cromwell_binary_list([[1,0,1,0,0,1,0],[0,0,0,1,0,0,1],[0,0,1,1,0,0,0],[1,1,0,0,1,0,0],[0,0,0,0,1,0,1],[0,1,0,0,0,1,0]]))
print(cromwell_binary_list([[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,0,1],[0,1,0,1,1,0],[1,0,0,0,1,1]]))
print(cromwell_binary_list([[1,1,0,0],[1,0,1,1],[0,1,1,1]]))
print(cromwell_binary_list([[1,0,0,1,0],[1,0,0,0,1],[0,1,1,0,1],[0,1,1,1,0]]))
print(cromwell_binary_list([[0,0,0,1,1,0],[0,0,1,1,0,1],[1,0,0,0,1,0],[1,1,1,0,0,0],[0,1,0,0,0,1]]))

print("="*30)

print(R2([82, 9, 24, 100, 5, 34])) #false
print(R3([82, 9, 24, 100, 5, 34])) #false
print(R2([40, 20, 9, 22, 35])) #true
print(R3([40, 20, 9, 22, 35])) #true

print("="*30)

print(R2([12, 11, 7])) #false
print(R3([12, 11, 7])) #false
print(R2([18, 17, 13, 14])) #true
print(R3([18, 17, 13, 14])) #false

print("="*30)

print(R2([12,11,7])) #false
print(R3([12,11,7])) #false
print(R2([6, 13, 34, 56, 17])) #true
print(R3([6, 13, 34, 56, 17])) #true

print("="*30)

print(R2([7,7])) #false
print(R3([7,7])) #false
print(R2([14, 17, 9, 22])) #false
print(R3([14, 17, 9, 22])) #false