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
    
class Poly():
    def __init__(self, terms = None):
        self.terms = {}
        if terms:
            for deg, coef in terms.items():
                if coef != 0:
                    self.terms[int(deg)] = coef
        if self.terms:
            self.maxDegree = max(self.terms.keys())
            self.minDegree = min(self.terms.keys())
        else:
            self.maxDegree = self.minDegree = 0
 
    def __repr__(self):
        if not self.terms:
            return "0"
        terms_str = []
        for deg in sorted(self.terms.keys(), reverse=True):
            coef = self.terms[deg]
            if deg == 0:
                terms_str.append(f"{coef}")
            elif deg == 1:
                terms_str.append(f"{coef}x")
            else:
                terms_str.append(f"{coef}x^{deg}")
        return " + ".join(terms_str).replace("+ -", "- ")
    
    __str__ = __repr__

    def isZero(self):
        return not self.terms

    def __add__(self, other):
        result = self.terms.copy()
        for deg, coef in other.terms.items():
            result[deg] = result.get(deg, 0) + coef
            if result[deg] == 0:
                del result[deg]
        return Poly(result)
    
    def __sub__(self, other):
        result = self.terms.copy()
        for deg, coef in other.terms.items():
            result[deg] = result.get(deg, 0) - coef
            if result[deg] == 0:
                del result[deg]
        return Poly(result)
    
    def __mul__(self, other):
        result = {}
        for deg1, coef1 in self.terms.items():
            for deg2, coef2 in other.terms.items():
                deg = deg1 + deg2
                result[deg] = result.get(deg, 0) + coef1 * coef2
                if result[deg] == 0:
                    del result[deg]
        return Poly(result)
    
    def __eq__(self, other):
        return self.terms == other.terms

    def __hash__(self):
        return hash(tuple(sorted(self.terms.items())))
    
    def copy(self):
        return Poly(self.terms.copy())

def makeCromwell_normal(n):
    result = Matrix()
    result.rows = n
    result.columns = n

    while True:    
        col_counts = [0] * n
        result.matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            remaining = [j for j in range(n) if col_counts[j] < 2]
            if len(remaining) < 2:
                break
            ones = random.sample([j for j in range(n) if col_counts[j] < 2], 2)
            for j in ones:
                result.matrix[i][j] = 1
                col_counts[j] += 1
        else:
            break
        continue

    for _ in range(100):
        r1, r2 = random.sample(range(n), 2)
        c1_candidates = [c for c in range(n) if result.matrix[r1][c] == 1 and result.matrix[r2][c] == 0]
        c2_candidates = [c for c in range(n) if result.matrix[r1][c] == 0 and result.matrix[r2][c] == 1]
        if c1_candidates and c2_candidates:
            c1 = random.choice(c1_candidates)
            c2 = random.choice(c2_candidates)
            result.matrix[r1][c1], result.matrix[r1][c2] = 0, 1
            result.matrix[r2][c1], result.matrix[r2][c2] = 1, 0

    return result

def makeCromwell_spatial(n):
    result = Matrix()
    result.rows = n
    result.columns = n+1

    while True:
        col_counts = [0] * (n+1)
        result.matrix = [[0 for _ in range(n+1)] for _ in range(n)]
        for i in range(2):
            ones = random.sample(range(n+1), 3)
            for j in ones:
                result.matrix[i][j] = 1
                col_counts[j] += 1
        for i in range(2, n):
            remaining = [j for j in range(n+1) if col_counts[j] < 2]
            if len(remaining) < 2:
                break
            ones = random.sample([j for j in range(n+1) if col_counts[j] < 2], 2)
            for j in ones:
                result.matrix[i][j] = 1
                col_counts[j] += 1
        else:
            break
        continue

    random.shuffle(result.matrix)

    for _ in range(100):
        r1, r2 = random.sample(range(n), 2)
        c1_candidates = [c for c in range(n) if result.matrix[r1][c] == 1 and result.matrix[r2][c] == 0]
        c2_candidates = [c for c in range(n) if result.matrix[r1][c] == 0 and result.matrix[r2][c] == 1]
        if c1_candidates and c2_candidates:
            c1 = random.choice(c1_candidates)
            c2 = random.choice(c2_candidates)
            result.matrix[r1][c1], result.matrix[r1][c2] = 0, 1
            result.matrix[r2][c1], result.matrix[r2][c2] = 1, 0

    visited = [[False] * (n+1) for _ in range(n)]
    def bfs(start_row, start_col):
        q = deque([(start_row, start_col)])
        visited[start_row][start_col] = True
        while q:
            r, c = q.popleft()
            same_row = [i for i in range(n+1) if result.matrix[r][i] == 1 and not visited[r][i]]
            same_col = [i for i in range(n) if result.matrix[i][c] == 1 and not visited[i][c]]
            for nc in same_row:
                visited[r][nc] = True
                q.append((r, nc))
            for nr in same_col:
                visited[nr][c] = True
                q.append((nr, c))

    components = 0
    while True:
        breaked = False
        for i in range(n):
            for j in range(n+1):
                if result.matrix[i][j] == 1 and not visited[i][j]:
                    components += 1
                    bfs(i, j)
                    breaked = True
                    break
            if breaked:
                break
        else:
            break

    if components >= 2:
        result = makeCromwell_spatial(n)
        return result
    else:
        return result

def isThetaOrHandcuff(cromwell):
    n = cromwell.rows
    copy_cromwell = Matrix()
    copy_cromwell.rows = n
    copy_cromwell.columns = n+1
    copy_cromwell.matrix = [row[:] for row in cromwell.matrix]

    for row_i in range(n):
            if sum(cromwell.matrix[row_i]) == 3:
                three_row = list(filter(lambda x: cromwell.matrix[row_i][x] == 1, range(n+1)))
                new_matrix = Matrix()
                new_matrix.rows = n-1
                new_matrix.columns = n-1
                new_matrix.matrix = [[cromwell.matrix[i][j] for j in range(n+1) if j != three_row[0] and j != three_row[2]] for i in range(n) if i != row_i]
    
    if abs(new_matrix.det()) == 1:
        return "Theta"
    elif abs(new_matrix.det()) <= 2:
        return "Handcuff"
    else:
        print("망함망함개망함")
        raise "망함망함개망함"
    
n=5
row_list = ['111000', '101000', '100100', '100010', '100001', '010100', '010010', '010001', '001010', '001001', '000101']