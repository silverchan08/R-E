def cromwell_binary(cromwell):
    n = len(cromwell)
    binary_list = []
    for i in range(n):
        a=0
        for j in range(n+1):
            a += cromwell[i][j] * (2**(n+1-j))
        binary_list.append(a)
    return binary_list

print(cromwell_binary([[1,0,1],[0,0,1]]))