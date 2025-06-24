import copy

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
