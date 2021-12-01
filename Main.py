import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import os

# in 1
from subprocess import check_output

print(check_output(["ls", "C:/Users/sahil/Desktop/ScienceFair/input"]).decode("utf8"))
# in 2
# plot for stars at time 0
data = pd.read_csv("C:/Users/sahil/Desktop/ScienceFair/input/c_0000.csv")
data['t'] = 0
data['v'] = np.sqrt(data.vx ** 2 + data.vy ** 2 + data.vz ** 2)
data['Ec'] = 0.5 * data.m * (data.v) ** 2
data['r'] = np.sqrt(data.x ** 2 + data.y ** 2 + data.z ** 2)

for i in range(1, 19):
    if i < 10:
        step = "0" + str(i)
    else:
        step = str(i)
    file = "C:/Users/sahil/Desktop/ScienceFair/input/c_" + step + "00.csv"
    d = pd.read_csv(file)
    d['t'] = i * 100
    d['v'] = np.sqrt(d.vx ** 2 + d.vy ** 2 + d.vz ** 2)
    d['Ec'] = 0.5 * d.m * (d.v) ** 2
    d['r'] = np.sqrt(d.x ** 2 + d.y ** 2 + d.z ** 2)
    data = data.append(d)

# in 3/4

# Induvidual distribution graphs for each time interval
''''
for i in range (0, 1900, 100):
    plt.scatter(data[data.t==i].x,data[data.t==i].y,s=1,marker='+')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cluster on the xy plane at t=' + str(i))
    plt.grid()
    plt.show()
 
# stars at all time
plt.scatter(data.x,data.y,s=1,marker='+')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cluster on the xy plane at all time')
plt.grid()
plt.show()

#induvidual r/v distribution at time 0, 1800, and All Time
plt.hist(data[data.t==0].r,100,alpha=0.6)
plt.xlabel('r')
plt.title("Radius of stars from center at t=0")
plt.show()
plt.hist(data[data.t==0].v,100,alpha=0.6)
plt.xlabel('v')
plt.title("Velocity of stars from center at t=0")
plt.show()

sample = data[data.r<10]

plt.hist(sample[sample.t==1800].r,100,alpha=0.6)
plt.xlabel('r')
plt.title("Radius of stars from center at t=1800")
plt.show()
plt.hist(data[data.t==1800].v,100,alpha=0.6)
plt.xlabel('v')
plt.title("Velocity of stars from center at t=1800")
plt.show()


plt.hist(sample.r,100,alpha=0.6)
plt.xlabel('r')
plt.title("Radius of stars from center at all time")
plt.show()
plt.hist(data.v,100,alpha=0.6)
plt.xlabel('v')
plt.title("Velocity of stars from center at all time")
plt.show()

# other option
time=[0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800]
for i in time:
    sns.kdeplot(sample[sample.t==i].r,label=str(i))
plt.xlim(0,10)
plt.xlabel('r')
plt.title("Radius of stars from center at all individual times")
plt.show()

for i in time:
    sns.kdeplot(sample[sample.t==i].v,label=str(i))
plt.xlim(0,10)
plt.xlabel('v')
plt.title("Velocity of stars at all individual times")
plt.show()
#r/v put together

'''
plt.scatter(data[data.t==0].r,data[data.t==0].v,s=1,marker='+',color='blue',alpha=0.5,label='all stars')
plt.xlabel('r')
plt.ylabel('v')
plt.title('r/v distribution at t=0')
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# clustering begins

data0 = data.loc[:, data.columns != 'id']
data2 = data[data.t == 1800].iloc[:, [0, 3]]
plt.scatter(data2.x, data2.vx, color="blue")
plt.title("X vs Vx")
plt.xlabel("X of stars")
plt.ylabel("Vx of stars")
plt.show()
wcss = []
'''
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_)  # inertia means that find to value of wcss

plt.plot(range(1, 15), wcss)
plt.title("Finding cluster amount")
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()

# we can take elbow as 4
kmean2 = KMeans(n_clusters=4)
clusters = kmean2.fit_predict(data2)

data2["label"] = clusters

plt.scatter(data[data.t==1800].x[data2.label == 0], data[data.t==1800].y[data2.label == 0], color="red")
plt.scatter(data[data.t==1800].x[data2.label == 1], data[data.t==1800].y[data2.label == 1], color="blue")
plt.scatter(data[data.t==1800].x[data2.label == 2], data[data.t==1800].y[data2.label == 2], color="green")
plt.scatter(data[data.t==1800].x[data2.label == 3], data[data.t==1800].y[data2.label == 3], color="purple")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# histogram stuff

data3 = data2.iloc[:, data2.columns != 'label'].head(1000)
merg = linkage(data3, method="ward")  # scipy is an algorithm of hiyerarchal clusturing
dendrogram(merg, leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.title("Find amount of clusters for Vx")
plt.show()

hiyerartical_cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
cluster = hiyerartical_cluster.fit_predict(data3)

data3["label"] = cluster

plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data3.label == 0],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data3.label == 0], color="red")
plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data3.label == 1],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data3.label == 1], color="blue")
plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data3.label == 2],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data3.label == 2], color="green")
plt.title("X vs Y distribution with Vx clusters")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# vy

data4 = data[data.t == 1800].iloc[:, [1, 4]]
data5 = data4.iloc[:, data4.columns != 'label'].head(1000)
plt.scatter(data4.y, data4.vy, color="blue")
plt.title("Y vs Vy")
plt.xlabel("Y of stars")
plt.ylabel("Vy of stars")
plt.show()

merg = linkage(data3, method="ward")  # scipy is an algorithm of hiyerarchal clusturing
dendrogram(merg, leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.title("Find amount of clusters for Vy")
plt.show()

hiyerartical_cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
cluster = hiyerartical_cluster.fit_predict(data5)

data5["label"] = cluster

plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data5.label == 0],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data5.label == 0], color="red")
plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data5.label == 1],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data5.label == 1], color="blue")
plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data5.label == 2],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data5.label == 2], color="green")
plt.title("X vs Y distribution with Vy clusters")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data5.label == 0],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data5.label == 0], color="red")
plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data5.label == data3.label == 1],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data5.label == data3.label == 1],
            color="blue")
plt.scatter(data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).x[data5.label == 2],
            data[data.t == 1800].iloc[:, data.columns != 'label'].head(1000).y[data5.label == 2], color="red")
plt.title("X vs Y distribution with general clusters")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
'''