#! /usr/bin/env python3
import numpy as np
import pyspark
from pyspark import SparkContext

sc= pyspark.SparkContext("local","mle")
path ="./cyq/mledata.txt"
rdd1 = sc.textFile(path).cache()

def change(line):
    values = [float(x) for x in line.replace(',', ' ').split(' ')]
    return values 
test = rdd1.map(change)
theta2 = np.array((7.9663e-04,2.276092059e-02,1.7069186e-05))
theta = np.array((7.966334245595849e-09,2.2760920596717212e-09,1.7069186236076075e-09))
Lbest = test.map(lambda x: \
	x[-1]*np.log(1/(1+np.exp(-1*np.dot(x[:-1],theta))))+ \
	(1-x[-1])*np.log(1-(1/(1+np.exp(-1*np.dot(x[:-1],theta))))))
Lcomepare=test.map(lambda x: \
	x[-1]*np.log(1/(1+np.exp(-1*np.dot(x[:-1],theta2))))+ \
	(1-x[-1])*np.log(1-(1/(1+np.exp(-1*np.dot(x[:-1],theta2))))))
print("Lbest——"+str(Lbest.collect()))
print("Lcomepare——"+str(Lcomepare.collect()))
