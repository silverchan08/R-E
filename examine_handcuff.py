from collections import deque
import time
import topoly
import copy

def toBinary(n):
    ret = ''
    while n>0:
        if n%2 == 0:
            ret += '0'
        else:
            ret += '1'
        n >>= 1
    return ret[::-1]

def determinant(matrix):
    n = len(matrix)
    for p in range(n):
        q = p
        for i in range(p, n):
            if abs(matrix[i][p]) > abs(matrix[q][p]):
                q = i
        matrix[p], matrix[q] = matrix[q], matrix[p]

        if matrix[p][p] == 0:
            return 0
        
        for i in range(p+1, n):
            c = matrix[i][p] / matrix[p][p]
            for j in range(p, n):
                matrix[i][j] -= c * matrix[p][j]
    det = 1
    for i in range(n):
        det *= matrix[i][i]
    return det

def isThetaOrHandcuff(cromwell_bit):
    # Theta를 만들고 component를 셀 때, theta/handcuff를 해당 component에서 먼저 세고 다 셌는데 방문 못 한 곳이 있으면 False, None을 return, 아니면 True, "(종류)"
    n = len(cromwell_bit)

    cromwell = [[0]*(n+1) for _ in range(n)]
    # cromwell matrix 구성
    for row_idx in range(n):
        row_bit = cromwell_bit[row_idx]
        col_idx = 0
        while row_bit > 0:
            if row_bit & 1:
                cromwell[row_idx][col_idx] = 1
            row_bit >>= 1
            col_idx += 1

    for row_i in range(n):
            if sum(cromwell[row_i]) == 3:
                three_row = list(filter(lambda x: cromwell[row_i][x] == 1, range(n+1)))
                new_matrix = [[cromwell[i][j] for j in range(n+1) if j != three_row[0] and j != three_row[2]] for i in range(n) if i != row_i]

    det = determinant(new_matrix)
    if abs(det) == 1:
        return "Theta"
    elif abs(det) <= 2:
        return "Handcuff"
    else:
        print("망함망함개망함")
        raise "망함망함개망함"

def isComposite1(cromwell_bit): # 1자로 연결된 composite
    n = len(cromwell_bit)
    # 한 행을 기준으로 해당 행을 포함하여 위에만 봤을 때 외톨이인 점이 2개 => composite
    for row_idx in range(n): # 행 선택
        cnt = 0
        for col_idx in range(n+1): # 열 돌면서 확인
            # 해당 행을 포함하여 위에 혼자 있는 점 개수
            Above = 0
            for another_row_idx in range(row_idx):
                if (cromwell_bit[another_row_idx] >> col_idx) & 1:
                    Above += 1
            if Above == 1:
                cnt += 1
        if cnt == 2:
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
    return CP[-1]

def R1(list):
    # 1이 3개짜리인 행 찾기
    three = []
    n = len(list)
    for row_i in range(n):
        if list[row_i].bit_count() == 3:
            three.append(row_i)
    
    # 1이 하나만 겹치는 두 행 찾기
    for i in range(n):
        for j in range(i+1,n):
            t = 0
            if (list[i]&list[j]).bit_count() == 1:
                M = max(((list[i]&list[j])^list[i]).bit_length(),((list[i]&list[j])^list[j]).bit_length())
                M2 = min(LeftOne(list[i]),LeftOne(list[j]))
                N = (list[i]&list[j]).bit_length()
                # 곂치는 1이 제일 왼쪽에 있는 경우
                if list[i].bit_length() == list[j].bit_length():
                    # 1이 2개있는 것이 3개짜리 행의 일부라면 넘김
                    if (M == ((list[i]&list[j])^list[i]).bit_length() and i in three) or (M == ((list[i]&list[j])^list[j]).bit_length() and j in three):
                        continue
                    # 1이 하나만 곂치는 두 행 사이의 공간에 다른 1이 있다면 넘김
                    else:
                        for k in range(1,j-i):
                            if (list[i+k] >> M-1).bit_count() != (list[i+k] >> N).bit_count():
                                t = 1
                                break
                    if t:
                        break
                    return True
                # 곂치는 1이 제일 오른쪽에 있는 경우
                elif list[i].bit_length()>((list[i]&list[j]).bit_length()) and list[j].bit_length()>((list[i]&list[j]).bit_length()):
                    # 1이 2개있는 것이 3개짜리 행의 일부라면 넘김
                    if i in three and (list[i] >> (list[j].bit_length())).bit_count() <= 1:
                        continue
                    elif j in three and (list[j] >> (list[i].bit_length())).bit_count() <= 1:
                        continue
                    # 1이 하나만 곂치는 두 행 사이의 공간에 다른 1이 있다면 넘김
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

def Theta_makeCromwell_spatial(n):
    if n == 2:
        return [[7, 7]]
    
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
        for k in range(j-1, 0, -1):
            first_three_ones_rows.append(1<<n | 1<<j | 1<<k)
    # 좌우대칭 제거: 000000이 있으면 110001에서 101001까지 0000000이 있으면 1100001에서 1001001까지
    for j in range(n-1, (n-1)//2, -1):
        first_three_ones_rows.append(1<<n | 1<<j | 1<<0)
    
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
                # 위아래 대칭 제거: 맨 아래에 1 세 개를 채울 때는 맨 위보다 큰 값만 넣기
                first_row_idx = three_ones_rows.index(current_matrix_rows[0])
                candidate_for_current_row = three_ones_rows[first_row_idx+1:]
            else:
                candidate_for_current_row = all_candidate_rows

        for candidate_row in candidate_for_current_row:
            candidate_row_idx = all_candidate_rows.index(candidate_row)

            isValid = True

            # 바로 위랑 겹치는 1이 있으면 쳐내기 (이때 해당 행과 바로 위 행은 1이 두 개여야 함)
            if current_matrix_rows[-1].bit_count() == 2 and candidate_row.bit_count() == 2:
                if current_matrix_rows[-1] & candidate_row:
                    continue

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

            # 현재 후보를 넣은 (bitmasked) matrix 생성
            Newone = current_matrix_rows[:]
            Newone[row_idx] = candidate_row

            # composite면 쳐내기
            if isComposite1(Newone):
                continue

            # 라이데마이스터 변환 1
            if R1(Newone):
                continue

            if row_idx == n-1:
                if R2(Newone):
                    continue

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

def isOneComponent(cromwell_bit):
    n = len(cromwell_bit)

    visited = [[False] * (n+1) for _ in range(n)]

    def bfs(start_row, start_col):
        q = deque([(start_row, start_col)])
        visited[start_row][start_col] = True
        while q:
            r, c = q.popleft()
            same_row = [i for i in range(n+1) if (cromwell_bit[r] >> i) & 1 and not visited[r][i]]
            same_col = [i for i in range(n) if (cromwell_bit[i] >> c) & 1 and not visited[i][c]]
            for nc in same_row:
                visited[r][nc] = True
                q.append((r, nc))
            for nr in same_col:
                visited[nr][c] = True
                q.append((nr, c))

    components = 0
    
    for i in range(n):
        for j in range(n+1):
            if (cromwell_bit[i] >> j) & 1 and not visited[i][j]:
                components += 1
                if components > 1:
                    return False
                bfs(i, j)
                break
    return True

def get_pd_code(cromwell_bit):
    n = len(cromwell_bit)
    
    # 코드 담는 리스트
    # vertex면 V, crossing면 X
    # V는 [왼쪽, 가운데, 오른쪽, 가운데의 방향]
    # X는 [(시계방향인 네 가닥), 아래 가닥의 방향]
    planar = []

    cromwell = [[0]*(n+1) for _ in range(n)]

    # cromwell matrix 구성
    for row_idx in range(n):
        row_bit = cromwell_bit[row_idx]
        col_idx = 0
        while row_bit > 0:
            if row_bit & 1:
                cromwell[row_idx][col_idx] = 1
            row_bit >>= 1
            col_idx += 1

    # 1이 세 개 있는 행과 그 중 가운데 1
    three_ones_rows = [row_idx for row_idx, row in enumerate(cromwell_bit) if row.bit_count() == 3]
    middle_three_ones_rows = []
    for row in three_ones_rows:
        cnt = 0
        for idx in range(n+1):
            if (cromwell_bit[row] >> idx) & 1:
                if cnt == 1:
                    middle_three_ones_rows.append(idx)
                    break
                cnt += 1

    # 각 점 저장
    first_three_vertex = [three_ones_rows[0], middle_three_ones_rows[0]]
    second_three_vertex = [three_ones_rows[1], middle_three_ones_rows[1]]

    # crossing 부분은 2로 저장
    for col_idx in range(n):
        # 해당 col의 두 1 사이만 봄
        col_ones = [one_row_idx for one_row_idx in range(n) if cromwell[one_row_idx][col_idx]]
        for crossing_row in range(col_ones[0]+1, col_ones[1]):
            # 해당 점이 해당 row의 1들 사이에 있으면 crossing => 2로 변경
            current_col_ones = [current_row_one for current_row_one in range(n+1) if cromwell[crossing_row][current_row_one] == 1]
            if current_col_ones[0] < col_idx < current_col_ones[-1]:
                cromwell[crossing_row][col_idx] = 2
    X = copy.deepcopy(cromwell)
    # n by n+1 방문 리스트, 이동 횟수, crossing에 붙일 숫자
    visited = [[False]*(n+1) for _ in range(n)]
    num_move = 1
    crossing_label = 1

    # planar가 다 추가됐는지 여부
    isFinished = False

    # 행 내 이동 / first_three_ones_row에서 시작
    def move_row_left(current_point):
        nonlocal num_move, crossing_label, visited, cromwell, cromwell_bit, planar, isFinished

        current_row = current_point[0]
        current_col = current_point[1]

        # 행 이동할 다음 점
        next_point = [current_row, 0]

        # 1이 세 개일 때
        if current_row == first_three_vertex[0] or current_row == second_three_vertex[0]:
            ones_same_row = [col_idx for col_idx in range(n+1) if (cromwell_bit[current_row] >> col_idx) & 1] # 같은 행에 있는 1들의 리스트
            # 맨 왼쪽/오른쪽 1일 때: 가운데로
            if current_col == ones_same_row[0] or current_col == ones_same_row[2]:
                next_point[1] = ones_same_row[1]
            # 가운데 1일 때: 안 간 곳으로
            else:
                if not visited[current_row][ones_same_row[0]]:
                    next_point[1] = ones_same_row[0]
                else:
                    next_point[1] = ones_same_row[2]
                # 만약 첫번째 1 세 개짜리 행인데 처음 지나가면 labeling
                if current_point == first_three_vertex:
                    if cromwell[current_row][current_col] == 1:
                        cromwell[current_row][current_col] += crossing_label + 1
                        crossing_label += 1
                        planar.append([num_move, 0, 0, "down"])
                # 두번째 1 세 개짜리 행이면 labeling
                else:
                    if not visited[current_row][ones_same_row[0]]:
                        planar[-1][0] = num_move
                    else:
                        planar[-1][2] = num_move
        # 1이 두 개일 때: 해당 행 다른 칸으로
        else:
            for col_idx in range(n+1):
                if cromwell[current_row][col_idx] == 1 and col_idx != current_col:
                    next_point[1] = col_idx
                    break

        # 왼쪽으로 이동한 경우
        if current_col > next_point[1]:
            # 그 사이 점들 탐색
            for middle_point_col in range(current_col-1, next_point[1]-1, -1):
                # 만약 0, 1이 아닌 경우: 2거나(교점) 5개거나(이미 지나가서 labeling한 교점) 3개(이미 지나간 1 세 개짜리 행)
                if cromwell[current_row][middle_point_col] not in [0, 1]:
                    if cromwell[current_row][middle_point_col] == 2: # 2일 때 (교점) => labeling
                        cromwell[current_row][middle_point_col] += crossing_label
                        planar.append([num_move, 0, num_move+1, 0, 'left'])
                        crossing_label += 1
                    elif len(planar[cromwell[current_row][middle_point_col]-3]) == 5: # 교점들은 3부터 labeling (3으로 labeling 돼있는 게 0번째)
                        vertex_pd_code = planar[cromwell[current_row][middle_point_col]-3]
                        vertex_pd_code[0] = num_move
                        vertex_pd_code[2] = num_move + 1
                        if vertex_pd_code[4] == 'up':
                            vertex_pd_code[1], vertex_pd_code[3] = vertex_pd_code[3], vertex_pd_code[1]
                    else: # 1 세 개짜리 행
                        vertex_pd_code = planar[cromwell[current_row][middle_point_col]-3]
                        if next_point == first_three_vertex:
                            vertex_pd_code[2] = num_move
                            vertex_pd_code[1] = num_move+1
                        else:
                            vertex_pd_code[2] = num_move
                    num_move += 1
                # vertex(cromwell이 1)인 경우
                elif cromwell[current_row][middle_point_col] == 1:
                    if [current_row, middle_point_col] == second_three_vertex:
                        cromwell[current_row][middle_point_col] += crossing_label+1
                        crossing_label += 1
                        planar.append([0, 0, num_move, "down"])
                        num_move += 1

        # 오른쪽으로 이동한 경우
        if current_col < next_point[1]:
            # 그 사이 점들 탐색
            for middle_point_col in range(current_col+1, next_point[1]+1):
                # 만약 0, 1이 아닌 경우: 2거나(교점) 5개거나(이미 지나가서 labeling한 교점) 3개(이미 지나간 1 세 개짜리 행)
                if cromwell[current_row][middle_point_col] not in [0, 1]:
                    if cromwell[current_row][middle_point_col] == 2: # 2일 때 (교점) => labeling
                        cromwell[current_row][middle_point_col] += crossing_label
                        planar.append([num_move, 0, num_move+1, 0, 'right'])
                        crossing_label += 1
                    elif len(planar[cromwell[current_row][middle_point_col]-3]) == 5: # 교점들은 3부터 labeling (3으로 labeling 돼있는 게 0번째)
                        vertex_pd_code = planar[cromwell[current_row][middle_point_col]-3]
                        vertex_pd_code[0] = num_move
                        vertex_pd_code[2] = num_move + 1
                        if vertex_pd_code[4] == 'down':
                            vertex_pd_code[1], vertex_pd_code[3] = vertex_pd_code[3], vertex_pd_code[1]
                    else: # 1 세 개짜리 행
                        vertex_pd_code = planar[cromwell[current_row][middle_point_col]-3]
                        if next_point == first_three_vertex:
                            vertex_pd_code[0] = num_move
                            vertex_pd_code[1] = num_move+1
                        else:
                            vertex_pd_code[0] = num_move
                    num_move += 1
                # vertex(cromwell이 1)인 경우
                elif cromwell[current_row][middle_point_col] == 1:
                    if [current_row, middle_point_col] == second_three_vertex:
                        cromwell[current_row][middle_point_col] += crossing_label+1
                        crossing_label += 1
                        planar.append([num_move, 0, 0, "down"])
                        num_move += 1

        if visited[next_point[0]][next_point[1]]:
            isFinished = True
            return
        visited[next_point[0]][next_point[1]] = True
        return next_point

    def move_row_right(current_point):
        nonlocal num_move, crossing_label, visited, cromwell, cromwell_bit, planar, isFinished

        current_row = current_point[0]
        current_col = current_point[1]
        # 행 이동할 다음 점
        next_point = [current_row, 0]

        # 1이 세 개일 때
        if current_row == first_three_vertex[0] or current_row == second_three_vertex[0]:
            ones_same_row = [col_idx for col_idx in range(n+1) if (cromwell_bit[current_row] >> col_idx) & 1] # 같은 행에 있는 1들의 리스트
            # 맨 왼쪽/오른쪽 1일 때: 가운데로
            if current_col == ones_same_row[0] or current_col == ones_same_row[2]:
                next_point[1] = ones_same_row[1]
            # 가운데 1일 때: 안 간 곳으로
            else:
                if not visited[current_row][ones_same_row[2]]:
                    next_point[1] = ones_same_row[2]
                else:
                    next_point[1] = ones_same_row[0]
                # 만약 첫번째 1 세 개짜리 행인데 처음 지나가면 labeling
                if current_point == first_three_vertex:
                    if cromwell[current_row][current_col] == 1:
                        cromwell[current_row][current_col] += crossing_label + 1
                        crossing_label += 1
                        planar.append([num_move, 0, 0, "down"])
                # 두번째 1 세 개짜리 행이면 labeling
                else:
                    if not visited[current_row][ones_same_row[2]]:
                        planar[-1][2] = num_move
                    else:
                        planar[-1][0] = num_move
        # 1이 두 개일 때: 해당 행 다른 칸으로
        else:
            for col_idx in range(n+1):
                if cromwell[current_row][col_idx] == 1 and col_idx != current_col:
                    next_point[1] = col_idx
                    break

        # 왼쪽으로 이동한 경우
        if current_col > next_point[1]:
            # 그 사이 점들 탐색
            for middle_point_col in range(current_col-1, next_point[1]-1, -1):
                # 만약 0, 1이 아닌 경우: 2거나(교점) 5개거나(이미 지나가서 labeling한 교점) 3개(이미 지나간 1 세 개짜리 행)
                if cromwell[current_row][middle_point_col] not in [0, 1]:
                    if cromwell[current_row][middle_point_col] == 2: # 2일 때 (교점) => labeling
                        cromwell[current_row][middle_point_col] += crossing_label
                        planar.append([num_move, 0, num_move+1, 0, 'left'])
                        crossing_label += 1
                    elif len(planar[cromwell[current_row][middle_point_col]-3]) == 5: # 교점들은 3부터 labeling (3으로 labeling 돼있는 게 0번째)
                        vertex_pd_code = planar[cromwell[current_row][middle_point_col]-3]
                        vertex_pd_code[0] = num_move
                        vertex_pd_code[2] = num_move + 1
                        if vertex_pd_code[4] == 'up':
                            vertex_pd_code[1], vertex_pd_code[3] = vertex_pd_code[3], vertex_pd_code[1]
                    else: # 1 세 개짜리 행
                        vertex_pd_code = planar[cromwell[current_row][middle_point_col]-3]
                        if next_point == first_three_vertex:
                            vertex_pd_code[2] = num_move
                            vertex_pd_code[1] = num_move+1
                        else:
                            vertex_pd_code[2] = num_move
                    num_move += 1
                # vertex(cromwell이 1)인 경우
                elif cromwell[current_row][middle_point_col] == 1:
                    if [current_row, middle_point_col] == second_three_vertex:
                        cromwell[current_row][middle_point_col] += crossing_label+1
                        crossing_label += 1
                        planar.append([0, 0, num_move, "down"])
                        num_move += 1

        # 오른쪽으로 이동한 경우
        if current_col < next_point[1]:
            # 그 사이 점들 탐색
            for middle_point_col in range(current_col+1, next_point[1]+1):
                # 만약 0, 1이 아닌 경우: 2거나(교점) 5개거나(이미 지나가서 labeling한 교점) 3개(이미 지나간 1 세 개짜리 행)
                if cromwell[current_row][middle_point_col] not in [0, 1]:
                    if cromwell[current_row][middle_point_col] == 2: # 2일 때 (교점) => labeling
                        cromwell[current_row][middle_point_col] += crossing_label
                        planar.append([num_move, 0, num_move+1, 0, 'right'])
                        crossing_label += 1
                    elif len(planar[cromwell[current_row][middle_point_col]-3]) == 5: # 교점들은 3부터 labeling (3으로 labeling 돼있는 게 0번째)
                        vertex_pd_code = planar[cromwell[current_row][middle_point_col]-3]
                        vertex_pd_code[0] = num_move
                        vertex_pd_code[2] = num_move + 1
                        if vertex_pd_code[4] == 'down':
                            vertex_pd_code[1], vertex_pd_code[3] = vertex_pd_code[3], vertex_pd_code[1]
                    else: # 1 세 개짜리 행
                        vertex_pd_code = planar[cromwell[current_row][middle_point_col]-3]
                        if next_point == first_three_vertex:
                            vertex_pd_code[0] = num_move
                            vertex_pd_code[1] = num_move+1
                        else:
                            vertex_pd_code[0] = num_move
                    num_move += 1
                # vertex(cromwell이 1)인 경우
                elif cromwell[current_row][middle_point_col] == 1:
                    if [current_row, middle_point_col] == second_three_vertex:
                        cromwell[current_row][middle_point_col] += crossing_label+1
                        crossing_label += 1
                        planar.append([num_move, 0, 0, "down"])
                        num_move += 1

        if visited[next_point[0]][next_point[1]]:
            isFinished = True
            return
        visited[next_point[0]][next_point[1]] = True
        return next_point

    def move_col(current_point):
        nonlocal num_move, crossing_label, visited, cromwell, cromwell_bit, planar, isFinished

        current_row = current_point[0]
        current_col = current_point[1]
        
        # 해당 열 다른 점으로 이동
        next_point = [0, current_col]
        for row_idx in range(n):
            if row_idx != current_row and ((cromwell_bit[row_idx] >> current_col) & 1):
                next_point[0] = row_idx
                break
        # 만약 첫번째 1 세 개짜리 행인데 처음 지나가면 labeling
        if current_point == first_three_vertex:
            if cromwell[current_row][current_col] == 1:
                cromwell[current_row][current_col] += crossing_label + 1
                crossing_label += 1
                planar.append([num_move, 0, 0, "down"])

        # 첫번째 1 세 개짜리 행이고 올라가면 마지막 up으로 변경 (기존 down)
        if (current_point == first_three_vertex):
            if current_row > next_point[0]:
                planar[0][3] = "up"

        # 두번째 1 세 개짜리 행이면 V 점의 두번째에 num_move 추가, 단 올라갈 땐 마지막 up으로 변경 (기존 down)
        if (current_point == second_three_vertex):
            planar[-1][1] = num_move
            if current_row > next_point[0]:
                planar[-1][3] = "down"
        
        # 위로 이동한 경우
        if current_row > next_point[0]:
            # 그 사이 점들 탐색
            for middle_point_row in range(current_row-1, next_point[0]-1, -1):
                # 만약 0, 1이 아닌 경우: 2거나(교점) 5개거나(이미 지나가서 labeling한 교점) 4개(이미 지나간 1 세 개짜리 행)
                if cromwell[middle_point_row][current_col] not in [0, 1]:
                    if cromwell[middle_point_row][current_col] == 2: # 2일 때 (교점) => labeling
                        cromwell[middle_point_row][current_col] += crossing_label
                        planar.append([0, num_move, 0, num_move+1, 'up'])
                        crossing_label += 1
                    elif len(planar[cromwell[middle_point_row][current_col]-3]) == 5: # 교점들은 3부터 labeling (3으로 labeling 돼있는 게 0번째)
                        vertex_pd_code = planar[cromwell[middle_point_row][current_col]-3]
                        vertex_pd_code[1] = num_move
                        vertex_pd_code[3] = num_move + 1
                        if vertex_pd_code[4] == 'left':
                            vertex_pd_code[1], vertex_pd_code[3] = vertex_pd_code[3], vertex_pd_code[1]
                    else: # 1 세 개짜리 행
                        vertex_pd_code = planar[cromwell[middle_point_row][current_col]-3]
                        if next_point == first_three_vertex:
                            vertex_pd_code[1] = num_move
                            vertex_pd_code[2] = num_move+1
                        else:
                            vertex_pd_code[1] = num_move
                        vertex_pd_code[3] = "down"
                    num_move += 1
                # vertex(cromwell이 1)인 경우
                elif cromwell[middle_point_row][current_col] == 1:
                    if [middle_point_row, current_col] == second_three_vertex:
                        cromwell[middle_point_row][current_col] += crossing_label+1
                        crossing_label += 1
                        planar.append([0, num_move, 0, "down"])
                        num_move += 1

        # 아래로 이동한 경우
        if current_row < next_point[0]:
            # 그 사이 점들 탐색
            for middle_point_row in range(current_row+1, next_point[0]+1):
                # 만약 0, 1이 아닌 경우: 2거나(교점) 5개거나(이미 지나가서 labeling한 교점) 4개(이미 지나간 1 세 개짜리 행)
                if cromwell[middle_point_row][current_col] not in [0, 1]:
                    if cromwell[middle_point_row][current_col] == 2: # 2일 때 (교점) => labeling
                        cromwell[middle_point_row][current_col] += crossing_label
                        planar.append([0, num_move, 0, num_move+1, 'down'])
                        crossing_label += 1
                    elif len(planar[cromwell[middle_point_row][current_col]-3]) == 5: # 교점들은 3부터 labeling (3으로 labeling 돼있는 게 0번째)
                        vertex_pd_code = planar[cromwell[middle_point_row][current_col]-3]
                        vertex_pd_code[1] = num_move
                        vertex_pd_code[3] = num_move + 1
                        if vertex_pd_code[4] == 'right':
                            vertex_pd_code[1], vertex_pd_code[3] = vertex_pd_code[3], vertex_pd_code[1]
                    else: # 1 세 개짜리 행
                        vertex_pd_code = planar[cromwell[middle_point_row][current_col]-3]
                        if next_point == first_three_vertex:
                            vertex_pd_code[1] = num_move
                            vertex_pd_code[2] = num_move+1
                        else:
                            vertex_pd_code[1] = num_move
                        vertex_pd_code[3] = "up"
                    num_move += 1
                # vertex(cromwell이 1)인 경우
                elif cromwell[middle_point_row][current_col] == 1:
                    if [middle_point_row, current_col] == second_three_vertex:
                        cromwell[middle_point_row][current_col] += crossing_label+1
                        crossing_label += 1
                        planar.append([0, num_move, 0, "up"])
                        num_move += 1

        if visited[next_point[0]][next_point[1]]:
            isFinished = True
            return
        visited[next_point[0]][next_point[1]] = True
        return next_point
    
    current_point = first_three_vertex
    while not isFinished:
        if not isFinished:
            current_point = move_row_left(current_point)
        if not isFinished:
            current_point = move_col(current_point)

    ret = ''
    for idx, elem in enumerate(planar):
        if len(elem) == 4:
            ret += f"V[{elem[0]}, {elem[1]}, {elem[2]}]" if elem[3] == "down" else f"V[{elem[0]}, {elem[2]}, {elem[1]}]"
        else:
            ret += f"X[{elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}]"
        ret += ';' if idx != len(planar)-1 else ''
    ret1 = ret

    isFinished = False  #변수 초기화
    current_point = first_three_vertex
    visited = [[0]*(n+1) for i in range(n)]
    num_move = 1
    crossing_label = 1
    cromwell = X
    planar = []
    current_point = first_three_vertex
    while not isFinished:   #오른 시작으로 한 번 더
        if not isFinished:
            current_point = move_row_right(current_point)
        if not isFinished:
            current_point = move_col(current_point)

    ret = ''
    for idx, elem in enumerate(planar):
        if len(elem) == 4:
            ret += f"V[{elem[0]}, {elem[1]}, {elem[2]}]" if elem[3] == "down" else f"V[{elem[0]}, {elem[2]}, {elem[1]}]"
        else:
            ret += f"X[{elem[0]}, {elem[1]}, {elem[2]}, {elem[3]}]"
        ret += ';' if idx != len(planar)-1 else ''
    ret2 = ret
    if ' 0' in ret1 or '[0' in ret1: #둘 중에 하나는 맞음, 틀린 것은 꼬리에 0 남음
        return ret2
    else:
        return ret1

cromwell = [73, 18, 132, 320, 34, 640, 264, 36, 529]
pd = get_pd_code(cromwell)
knot = topoly.yamada(pd)
print(knot, topoly.getpoly('y', knot))