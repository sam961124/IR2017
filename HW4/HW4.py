import numpy as np

doc_size = 1095
doc_vector = np.load('doc_vector.npy')
C = np.load('C.npy')
I = np.load('I.npy')


def Sim(Docx, Docy):
    return (doc_vector[Docx] * doc_vector[Docy]).sum()
def find_max_sim(C, I):
    max_sim = -1
    index_i = -1
    index_m = -1
    for i in range(doc_size):
        if I[i] == 1:
            for m in range(doc_size):
                if I[m] == 1 and m != i:
                    if max_sim < C[i][m]:
                        max_sim = C[i][m]
                        index_i = i
                        index_m = m
    return index_i, index_m



A = []
for k in range(doc_size - 1):
    i, m = find_max_sim(C, I)
    A.append([i ,m])
    for j in range(doc_size):
        C[i][j] = min(Sim(i, j), Sim(m, j))
        C[j][i] = min(Sim(j, i), Sim(j, m))
    I[m] = 0
    if k % 100 == 0:
        print(str(k*100//doc_size) + '%')




def write_cluster(cluster_dict, K):
    with open(str(K) + '.txt', 'w') as cluster_file:
        for key, l in cluster_dict.items():
            doc_list = np.sort(l)
            for doc_id in doc_list:
                cluster_file.write(str(doc_id+1) + '\n')
            cluster_file.write('\n')



cluster_dict = {}
for i in range(doc_size):
    cluster_dict[str(i)] = [i]
for i, m in A:
    temp = cluster_dict[str(m)]
    cluster_dict.pop(str(m), None)
    cluster_dict[str(i)] += temp
    if len(cluster_dict) == 20:
        write_cluster(cluster_dict, 20)
    elif len(cluster_dict) == 13:
        write_cluster(cluster_dict, 13)
    elif len(cluster_dict) == 8:
        write_cluster(cluster_dict, 8)

