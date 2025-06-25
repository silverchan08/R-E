import itertools
import time
import itertools

def Theta_makeCromwell_spatial(n):
    all_matrices = []

    # 1 두 개짜리 행 다 만들기 / 내림차순으로 생성
    two_ones_rows = []
    for i in range(n, -1, -1):
        for j in range(i-2, -1, -1): # 1이 연속으로 2개 있으면 수축 가능
            two_ones_rows.append(1<<i | 1<<j)

    # 1 세 개짜리 행 다 만들기 / 내림차순으로 생성
    three_ones_rows = []
    for i in range(n, -1, -1):
        for j in range(i-1, -1, -1):
            for k in range(j-1, -1, -1):
                three_ones_rows.append(1<<i | 1<<j | 1<<k)

    # 맨 위는 1xxx인 1 세 개짜리 행으로 고정 가능
    first_three_ones_rows = []
    for j in range(n-1, -1, -1):
        for k in range(j-1, -1, -1):
            first_three_ones_rows.append(1<<n | 1<<j | 1<<k)
    
    all_candidate_rows = sorted(two_ones_rows + three_ones_rows, reverse=True)

    three_rows_idx = []
    current_matrix_rows = [0]*n
    current_matrix_rows_idx = []
    current_column_ones = [0]*(n+1) # 열은 <- 방향으로 셈 (비트마스크에서 n번째가 n번째 열)

    def backtrack(row_idx):
        nonlocal current_matrix_rows, current_column_ones, all_matrices

        if row_idx == n: # 행 다 채웠을 때
            all_matrices.append(list(current_matrix_rows))
            return
        
        # 만약 현재 행이 0번째면 1xxx인 1 세 개짜리 행을 후보로, 아니면 현재까지 1 세 개짜리 행 개수에 따라 정함.
        # 이때 현재 행이 n-1번째인데 1 세 개짜리 행이 1개면 무조건 추가.
        if row_idx == 0:
            candidate_for_current_row = first_three_ones_rows
        else:
            if len(three_rows_idx) == 2:
                candidate_for_current_row = two_ones_rows
            elif row_idx == n-1:
                candidate_for_current_row = three_ones_rows
            else:
                candidate_for_current_row = all_candidate_rows

        for candidate_row_idx, candidate_row in enumerate(candidate_for_current_row):
            isValid = True

            # 같은 행이 있으면 쳐내기
            for idx in current_matrix_rows_idx:
                if idx == candidate_row_idx:
                    isValid = False
                    break
            if not isValid:
                continue
            
            # 만약 한 열에 1이 3개 이상이면 패스
            for col_idx in range(n+1):
                if (candidate_row >> col_idx) & 1:
                    if current_column_ones[col_idx] == 2:
                        isValid = False
                        break
            if not isValid:
                continue
            
            # 쳐내는건 여기에다가, 쳐내는건 isValid를 False로

            if isValid:
                # 만약 유효하면 현재 열에 저장
                current_matrix_rows[row_idx] = candidate_row
                current_matrix_rows_idx.append(candidate_row_idx)

                # 만약 1이 세 개면 three_rows_idx에 추가
                if candidate_row.bit_count() == 3:
                    three_rows_idx.append(row_idx)

                # 열마다 1 추가로 count
                for col_idx in range(n+1):
                    if (candidate_row >> col_idx) & 1:
                        current_column_ones[col_idx] += 1
                
                # 다음 행 backtracking
                backtrack(row_idx+1)

                # 다시 원 상태로 복구
                current_matrix_rows[row_idx] = 0
                current_matrix_rows_idx.pop()

                if candidate_row.bit_count() == 3:
                    three_rows_idx.pop()
                
                for col_idx in range(n+1):
                    if (candidate_row >> col_idx) & 1:
                        current_column_ones[col_idx] -= 1
    
    # backtracking 시작
    backtrack(0)

    return all_matrices

# --- 사용 예시 ---
n_value = 6 # 예를 들어 n=4 (4x5 행렬)

'''s = time.time()

all_matrices = solve_all_special_matrices_in_desc_order(n_value)'''

'''print(all_matrices)'''

'''print(f"Found {len(all_matrices)} total matrices (including row permutations) for n = {n_value}:")
ii = [100, 100, 100, 100]
for i, matrix in enumerate(all_matrices):
    if ii < matrix:
        print(ii, i, matrix)
        exit(0)
    ii = matrix
    print(f"\nMatrix {i+1}:")
    for row_int in matrix:
        binary_string = bin(row_int)[2:].zfill(n_value + 1)
        print(binary_string)
    
    print("Decimal representation:", matrix)
'''
'''e = time.time()

print(len(all_matrices), e-s)
print(all_matrices == sorted(all_matrices, reverse=True))'''



s = time.time()

all_matrices = Theta_makeCromwell_spatial(n_value)

e = time.time()

print(len(all_matrices))
print(e-s)