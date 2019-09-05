#heavily inspired by Archetypal Analysis in R.
#Read this paper to compare the algorithm
#https://cran.r-project.org/web/packages/archetypes/vignettes/archetypes.pdf
#it's not that clean yet, but it works.

import random
import numpy as np
import scipy.optimize.nnls as nnls
import matplotlib.pyplot as plt
import copy

def plot_the_graph(name,data,matrix_z):
    x = []
    y = []
    for t in range(np.shape(matrix_z)[1]):
            x.append(matrix_z[0][t])
            y.append(matrix_z[1][t])
    x.append(x[0])
    y.append(y[0])

    plt.plot(data.transpose()[0],data.transpose()[1],'xb')
    #plt.plot(matrix_z[0],matrix_z[1],'Dr-')
    plt.plot(x,y,'Dr-')
    plt.savefig(name)
    plt.close()

#(data,group) = make_blobs(n_samples=samples, n_features=features, centers=center, cluster_std=0.8, random_state=0)
def my_least_square(A,b):
    X = A.transpose().dot(A)
    Y = A.transpose().dot(b)
    point = np.linalg.inv(X).dot(Y)


def init_matrix(row_size, column_size,column = True):
    matrix = np.zeros((row_size, column_size))
    if not column:
        matrix = matrix.transpose()
    for i in range(np.shape(matrix)[0]):
        row_sum = 0.0
        upper_bound = 1.0
        for c in range(np.shape(matrix)[1]):
            if c == np.shape(matrix)[1]-1:
                matrix[i][c] = upper_bound
            else:
                rand_num = random.uniform(0,upper_bound)
                matrix[i][c] = rand_num
                upper_bound -= rand_num
                row_sum += rand_num
    if not column:
        matrix = matrix.transpose()
    return matrix

samples = 20
features  = 2
center = 3
#you can choose any data
#data = np.random.randint(0,250, (samples,2))
#data = np.array([[80, 90,80,70, 100,80,60, 110,100,90,80,50, 120,60,40, 130,100,80,50,30],[10, 20,20,20, 30,30,30, 40,40,40,40,40, 50,50,50, 60,60,60,60,60]]).transpose()
#data = np.array([[10, 20,20,20, 30,30,30, 40,40,40,40,40, 50,50,50, 60,60,60,60,60],[80, 90,80,70, 100,80,60, 110,100,90,80,50, 120,60,40, 130,100,80,50,30]]).transpose()
data = np.array([[10, 20,20,20, 30,30,30, 40,50,40,50,40, 50,50,50, 60,60,60,60,60],[80, 90,80,70, 100,80,60, 110,100,90,80,50, 120,60,40, 130,100,80,50,30]]).transpose()
archaetype_size = 3

beta = init_matrix(archaetype_size,samples,False)
matrix_z = np.random.randint(13,130,(features,archaetype_size))
debug = copy.deepcopy(matrix_z)
#matrix_z = np.array([[30,50,50],[80,60,100]])

alpha = np.zeros((samples,center))

#data plotting
plot_the_graph("01.png",data,matrix_z)
#the first archetype

counter = 1
rss = 10000000
difference = 10
while counter != 100001:
    #find the best alphas
    for i in range(np.shape(data)[0]):
        a1,res = nnls(0.5*matrix_z,0.5*data[i])
        sum = np.sum(a1)
        a1 = a1/sum
        alpha[i] = a1

    #find the new archaetype
    alpha_pinv = np.linalg.pinv(alpha)
    new_matrix_z = alpha_pinv.dot(data).transpose()

    #find new beta
    new_beta = beta
    for j in range(np.shape(new_matrix_z)[1]):
        b1,res = nnls(data.transpose(),new_matrix_z.transpose()[j])
        sum = np.sum(b1)
        b1 = b1/sum
        #print(b1)
        new_beta[j] = b1
    new_beta = beta.transpose()

    #find new_archetype
    new_matrix_z = data.transpose().dot(new_beta)
    #print(new_matrix_z)

    matrix_z = new_matrix_z
    beta = new_beta.transpose()
    difference = rss - np.sum(np.square( data - alpha.dot(new_matrix_z.transpose()) ))
    rss = np.sum(np.square( data - alpha.dot(new_matrix_z.transpose()) ))
    print("{0} RSS = {1}".format(counter, rss))
    counter += 1

plot_the_graph("02.png",data,matrix_z)
