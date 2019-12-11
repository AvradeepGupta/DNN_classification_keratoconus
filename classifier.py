import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
import csv

labels = pd.read_csv('labels.csv')
#print(labels)
#print(labels.head())
Y = np.array(labels['clster_labels'])
#print(Y)

dataset = pd.read_csv('dataset.csv')
#print(dataset.head())
X = np.array(dataset[["Ks","Ks.Axis","Kf","Kf.Axis","AvgK","CYL","AA","Ecc.9.0mm.","ACCP","Ks.1","Ks.Axis.1","Kf.1","Kf.Axis.1","AvgK.1","CYL.1","AA.1","Ecc.9.0mm..1","Ks.2","Ks.Axis.2","Kf.2","Kf.Axis.2","AvgK.2","CYL.2","AA.2","Ecc.9.0mm..2","Ks.3","Ks.Axis.3","Kf.3","Kf.Axis.3","AvgK.3","CYL.3","AA.3","Apex","Thinnest","LocationX","LocationY","Spherical","Regular","Regular.Axis","Asymmetry","Asymmetry.Axis","HigherOrder","Spherical.1","Regular.1","Regular.Axis.1","Asymmetry.1","Asymmetry.Axis.1","HigherOrder.1","Spherical.2","Regular.2","Regular.Axis.2","Asymmetry.2","Asymmetry.Axis.2","HigherOrder.2","Spherical.3","Regular.3","Regular.Axis.3","Asymmetry.3","Asymmetry.Axis.3","HigherOrder.3","Spherical.4","Regular.4","Regular.Axis.4","Asymmetry.4","Asymmetry.Axis.4","HigherOrder.4","Spherical.5","Regular.5","Regular.Axis.5","Asymmetry.5","Asymmetry.Axis.5","HigherOrder.5","Spherical.6","Regular.6","Regular.Axis.6","Asymmetry.6","Asymmetry.Axis.6","HigherOrder.6","Spherical.7","Regular.7","Regular.Axis.7","Asymmetry.7","Asymmetry.Axis.7","HigherOrder.7","X","Y","X.1","Y.1","X.2","Y.2","ApexK","Ks.0mm.","Ks.0mm..Axis","Kf.0mm.","Kf.0mm..Axis","AvgK.0mm.","CYL.0mm.","Ks.6mm.","Ks.6mm..Axis","Kf.6mm.","Kf.6mm..Axis","AvgK.6mm.","CYL.6mm.","DSI.4mm.","OSI.4mm.","MS.Axis.4mm.","DSI.5mm.","OSI.5mm.","MS.Axis.5mm.","DSI.6mm.","OSI.6mm.","MS.Axis.6mm.","DSI.9mm.","OSI.9mm.","MS.Axis.9mm.","CSI","SD_P.4mm.","CV_P.4mm.","ACP.3mm.","SAI","K.Max..8mm.","X.3","Y.3","K.Max..10mm.","X.4","Y.4","ApexK.1","Ks.0mm..1","Ks.0mm..Axis.1","Kf.0mm..1","Kf.0mm..Axis.1","AvgK.0mm..1","CYL.0mm..1","Ks.6mm..1","Ks.6mm..Axis.1","Kf.6mm..1","Kf.6mm..Axis.1","AvgK.6mm..1","CYL.6mm..1","DSI.4mm..1","OSI.4mm..1","MS.Axis.4mm..1","DSI.5mm..1","OSI.5mm..1","MS.Axis.5mm..1","DSI.6mm..1","OSI.6mm..1","MS.Axis.6mm..1","DSI.9mm..1","OSI.9mm..1","MS.Axis.9mm..1","CSI.1","SD_P.4mm..1","CV_P.4mm..1","ACP.3mm..1","SAI.1","K.Max..8mm..1","X.5","Y.5","K.Max..10mm..1","X.6","Y.6","ApexK.2","Ks.0mm..2","Ks.0mm..Axis.2","Kf.0mm..2","Kf.0mm..Axis.2","AvgK.0mm..2","CYL.0mm..2","Ks.6mm..2","Ks.6mm..Axis.2","Kf.6mm..2","Kf.6mm..Axis.2","AvgK.6mm..2","CYL.6mm..2","DSI.4mm..2","OSI.4mm..2","MS.Axis.4mm..2","DSI.5mm..2","OSI.5mm..2","MS.Axis.5mm..2","DSI.6mm..2","OSI.6mm..2","MS.Axis.6mm..2","DSI.9mm..2","OSI.9mm..2","MS.Axis.9mm..2","CSI.2","SD_P.4mm..2","CV_P.4mm..2","ACP.3mm..2","SAI.2","K.Max..8mm..2","X.7","Y.7","K.Max..10mm..2","X.8","Y.8","Steepest","X.9","Y.9","Steepest.1","X.10","Y.10","Diameter","BFS_R","BFS_Ecc","BFS_OffX","BFS_OffY","BFS_OffZ","Highest.0mm.3mm.","X.11","Y.11","Highest.3mm.6mm.","X.12","Y.12","Highest.6mm.9mm.","X.13","Y.13","Highest.0mm.5mm.","X.14","Y.14","RMS_E.3mm.","RMS_E.4mm.","RMS_E.5mm.","RMS_E.6mm.","SR_E.4mm.","BFS_R.1","BFS_Ecc.1","BFS_OffX.1","BFS_OffY.1","BFS_OffZ.1","Highest.0mm.3mm..1","X.15","Y.15","Highest.3mm.6mm..1","X.16","Y.16","Highest.6mm.9mm..1","X.17","Y.17","Highest.0mm.5mm..1","X.18","Y.18","RMS_E.3mm..1","RMS_E.4mm..1","RMS_E.5mm..1","RMS_E.6mm..1","SR_E.4mm..1","Ecc.3mm.","Ecc.4mm.","Ecc.5mm.","Ecc.6mm.","Ecc.7mm.","Ecc.8mm.","Ecc.9mm.","Ecc.12mm.","Ecc.15mm.","SR_H.3mm.","SR_H.4mm.","SR_H.5mm.","SR_H.6mm.","Avg_H.5mm.","Avg_H.6mm.","Avg_H.7mm.","Avg_H.8mm.","Avg_H.9mm.","Avg_H.10mm.","Ecc.3mm..1","Ecc.4mm..1","Ecc.5mm..1","Ecc.6mm..1","Ecc.7mm..1","Ecc.8mm..1","Ecc.9mm..1","Ecc.12mm..1","Ecc.15mm..1","SR_H.3mm..1","SR_H.4mm..1","SR_H.5mm..1","SR_H.6mm..1","Avg_H.5mm..1","Avg_H.6mm..1","Avg_H.7mm..1","Avg_H.8mm..1","Avg_H.9mm..1","Avg_H.10mm..1","CSI_T","SD_T.4mm.","SD_T.5mm.","SD_T.6mm.","CV_T.4mm.","CV_T.5mm.","CV_T.6mm.","PSI_SD.6mm.","PSI_CV.6mm.","Avg_T.6mm.","Thickest","LocationX.1","LocationY.1","Apex.1","Thinnest.4mm.","LocationX.2","LocationY.2","CSI_T.1","SD_T.4mm..1","CV_T.4mm..1","Apex.2","Thinnest.4mm..1","LocationX.3","LocationY.3","CSI_T.2","SD_T.4mm..2","CV_T.4mm..2","RMS_E.4mm..2","SR_E.4mm..2","Sph..Keratometric.","Reg..Keratometric.","Asy..Keratometric.","Hio..Keratometric.","Sph..Posterior.","Reg..Posterior.","Asy..Posterior.","Hio..Posterior.","Steepest.Posterior.","Thinnest.1","Score.Anterior.","AA.6mm..Anterior.","Score.Posterior.","AA.6mm..Posterior.","BFS.Angle0.360.Dia4.9.Float.","X.19","Y.19","Z","LD8.5","LD8.8","LD9.4","Height.Avg..Angle0.360.Dia8.FlatRate0.","R.Angle0.360.Dia8.FlatRate0.","Ks.4","Kf.4","AA.4","DSI","OSI","CSI.3","KPI","KCI","Result","MaxElv","MinElv","RangeElv","Result.1","MaxElv.1","MinElv.1","RangeElv.1","Result.2","Steepest.2","Result.3","Steepest.3","Result.4","LabelNum","Steepest.4","OffX","OffY","P_Contact","Area","AvgRc","Result.5","LabelNum.1","Steepest.5","OffX.1","OffY.1","P_Contact.1","Area.1","AvgRc.1","Result.6","Apex.3","X0deg","X45deg","X90deg","X135deg","X180deg","X225deg","X270deg","X315deg","X0deg.1","X45deg.1","X90deg.1","X135deg.1","X180deg.1","BFS.1mm.","BFS.2mm.","BFS.3mm.","BFS.4mm.","BFS.5mm.","BFS.6mm.","BFS.7mm.","BFS.8mm.","BFS.1mm..1","BFS.2mm..1","BFS.3mm..1","BFS.4mm..1","BFS.5mm..1","BFS.6mm..1","BFS.7mm..1","BFS.8mm..1","coma","coma.axis","SA.C40.","S35.coma.like.","S46.sph..like.","HOAs.S3456.","coma.1","coma.axis.1","SA.C40..1","S35.coma.like..1","S46.sph..like..1","HOAs.S3456..1","coma.2","coma.axis.2","SA.C40..2","S35.coma.like..2","S46.sph..like..2","HOAs.S3456..2","coma.3","coma.axis.3","SA.C40..3","S35.coma.like..3","S46.sph..like..3","HOAs.S3456..3","coma.4","coma.axis.4","SA.C40..4","S35.coma.like..4","S46.sph..like..4","HOAs.S3456..4","coma.5","coma.axis.5","SA.C40..5","S35.coma.like..5","S46.sph..like..5","HOAs.S3456..5","AA.5","ESI.Anterior.","ESI.Posterior."]])
#print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=.6)

model = svm.SVC()
model.fit(X_train,Y_train)
see = model.predict(X_train[0].reshape(1,-1))
#print(see)
accuracy = model.score(X_test,Y_test)
print(accuracy*100)
# for i in Y_test:
#     print(i)

#X_test,Y_test,c = np.loadtxt('ex2data1.txt',delimiter=',', unpack=True)
plt.scatter(X_test[:,2],Y_test)
plt.show()

