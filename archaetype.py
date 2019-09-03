import random, numpy as np
#from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import scipy.optimize.nnls as nnls
import matplotlib.pyplot as plt

#(data,group) = make_blobs(n_samples=samples, n_features=features, centers=center, cluster_std=0.8, random_state=0)
def my_least_square(A,b):
    X = A.transpose().dot(A)
    Y = A.transpose().dot(b)
    point = np.linalg.inv(X).dot(Y)


def init_matrix(column_size, row_size):
    matrix = np.zeros((row_size, column_size))
    for i in range(row_size):
        row_sum = 0.0
        upper_bound = 1.0
        for c in range(column_size):
            if c == column_size-1:
                matrix[i][c] = 1.0 - row_sum
            else:
                rand_num = round(random.uniform(0,upper_bound), 2)
                matrix[i][c] = rand_num
                upper_bound -= rand_num
                row_sum += rand_num
    return matrix

samples = 100
features  = 2
center = 3
data = np.random.randint(0,250, (samples,2))
archaetype_size = 3

beta = init_matrix(archaetype_size,samples)
matrix_z = data.transpose().dot(beta)
alpha = np.zeros((samples,center))

#data plotting
plt.plot(data.transpose()[0],data.transpose()[1],'xb')
plt.plot(matrix_z[0],matrix_z[1],'Dr')
plt.savefig("my_data.png")
plt.close() #the first archetype

#find the best alphas
for i in range(np.shape(data)[0]):
    a1,res = nnls(0.5*matrix_z,0.5*data[i])
    sum = np.sum(y1)
    a1 = a1/sum
    alpha[i] = a1

#find the new archaetype
alpha_pinv = np.linalg.pinv(alpha)
new_matrix_z = alpha_pinv.dot(data).transpose()
#beta = beta.transpose()

for j in range(np.shape(new_matrix_z)[1]):
    b1,res = nnls(0.5*data.transpose(),0.5*new_matrix_z.transpose()[j])
    sum = np.sum(b1)
    b1 = b1/sum
    beta[i]= b1
beta = beta.transpose()






