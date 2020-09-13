#This program generates a shelve of D dimensional square matrices that can be used in ratios.py
import shelve
import numpy as np

while True:
    try:
        num_of_mat = input("Enter the number of the matrices required in the shelve: ")
        num_of_mat = int(num_of_mat)
        break
    except ValueError:
        print("Please select an integer value for the number of matrices")

while True:
    try:
        D = input("Enter the dimension of the matrices: ")
        D = int(D)
        break
    except ValueError:
        print("Please select an integer value for the dimension")

mat_dic = {}
for number in range(0, num_of_mat):
    mat_dic["mat" + str(number + 1)] = np.random.rand(D, D)

shelve_dict = shelve.open('dim-%d-mat-%d' % (D, num_of_mat), flag='n')
for i in mat_dic.keys():
    shelve_dict[i] = mat_dic[i]
shelve_dict.close()
