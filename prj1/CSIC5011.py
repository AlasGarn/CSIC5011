# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 18:03:22 2017

@author: sepanta
"""

print(__doc__)
from numpy import *
from math import sqrt
import numpy as np
import pandas
from sklearn import manifold
import mdtraj
import matplotlib.pyplot as plt
from scipy.linalg import svd
import matplotlib.ticker as ticker


def fmt(x, pos):
    a, b = '{:.4e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)



def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)
    
    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB
    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return R, t

def get_rmse(A, B):
    n = len(A)
    A = mat(A)
    B = mat(B)
    
    # recover the transformation
    ret_R, ret_t = rigid_transform_3D(A, B)
    A2 = (ret_R*A.T) + tile(ret_t, (1, n))
    A2 = A2.T
    
    # Find the error
    err = A2 - B
    err=err*err.T
    err = np.diag(err)
    err = np.sum(err)
    rmse = sqrt(err/n);
    return rmse


def get_Clean_Data(data):
    data.loc[1].values[0].split()
    cleanData = []
    for index, row in data.iterrows():
        print(row.values[0].split()[2:])    
        cleanData.append(row.values[0].split()[2:])
        
    cleanData= [map(float, row) for row in cleanData]  
    return cleanData
    
# write trajectory
def write_traj(pdb_iso, file_name):
    protein = {"ns":[],"xyzs":[]}
    i = 0
    for row in pdb_iso:
        protein["ns"].append(281+i)
        i = i+1
        protein["xyzs"].append(map(float,row))
    with open(file_name,"a") as f:
        f.write(str(len(protein["ns"])) + "\n\n")
        xyz = [" ".join(map(str, [np.round(x,3) for x in xyz])) for xyz in protein["xyzs"]]
        ns =  map(str, protein["ns"])
        lines = ["C" + " " + b for a, b in zip(ns, xyz)]
        f.write("\n".join(lines) + "\n")
    return

def do_experiments(pdb, dataset_name, sampleCount):
    #2d-embeding without noise
    pdb_iso = manifold.Isomap(n_neighbors=20, n_components=2).fit_transform(pdb)                                        
    pdb_lle = manifold.LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='standard').fit_transform(pdb)
    pandas.DataFrame(pdb_iso).to_csv("C:\Users\sepanta\Documents\University\CSIC5011\protein3D\\"+dataset_name+"\iso_n20Noise0D2.csv", header=False, index=False)
    pandas.DataFrame(pdb_lle).to_csv("C:\Users\sepanta\Documents\University\CSIC5011\protein3D\\"+dataset_name+"\\lle_n20Noise0D2.csv", header=False, index=False)

    #changing number of neighbours    
    for i in [5*x for x in range(1, len(pdb)/5)]:
        #with normal noise
        for sd in [2*x for x in range(0, 11)]:
            for counter in range(0, SAMPLE_COUNT):#To average over different random noise values
                if sd == 0:
                    pdbNoisy = pdb 
                else:
                    noise = np.random.normal(0, sd, (len(pdb), 3))
                    pdbNoisy = pdb + noise
                pdb_iso = manifold.Isomap(n_neighbors=i, n_components=3).fit_transform(pdbNoisy)                                        
                pdb_lle = manifold.LocallyLinearEmbedding(n_neighbors=20, n_components=3, method='standard').fit_transform(pdbNoisy)
                write_traj(pdb_iso, dataset_name+"\\Isomap\\NormalNoise\\n"+str(i)+"Noise"+str(sd)+".xyz")
                write_traj(pdb_lle, dataset_name+"\\LLE\\NormalNoise\\n"+str(i)+"Noise"+str(sd)+".xyz")
    
        #with uniform noise
        for max_val in [2*x for x in range(0, 11)]:
            for counter in range(0, SAMPLE_COUNT):#To average over different random noise values
                if max_val == 0:
                    pdbNoisy = pdb 
                else:
                    noise = np.random.uniform(-1*max_val, max_val, (len(pdb), 3))
                    pdbNoisy = pdb + noise
                pdb_iso = manifold.Isomap(n_neighbors=i, n_components=3).fit_transform(pdbNoisy)                                        
                pdb_lle = manifold.LocallyLinearEmbedding(n_neighbors=20, n_components=3, method='standard').fit_transform(pdbNoisy)
                write_traj(pdb_iso, dataset_name+"\\Isomap\\UniformNoise\\n"+str(i)+"Noise"+str(max_val)+".xyz")
                write_traj(pdb_lle, dataset_name+"\\LLE\\UniformNoise\\n"+str(i)+"Noise"+str(max_val)+".xyz")


def align_calc(X, Y):
    
    """
    X and Y are 3d arrays of float,
    must have the same number of entries.
    
    Returns RMSD calculated after centroid centering
    and optimal rotation calculated by Kabsch algorithm
    """
    
    assert X.shape == Y.shape, "X and Y must have the same number of entries!"
    
    X -= X.mean(axis = 0)
    Y -= Y.mean(axis = 0)
    if np.allclose(X,Y):
        return 0
    R = np.dot(Y.T, X)
    V, s, WT = svd(R, full_matrices=False)
    W = WT.T
    d = np.linalg.det(np.dot(W, V.T))
    Z = np.eye(3)
    Z[2, 2] = d
    U  = np.dot(W, np.dot(Z, V.T))
    Ytr = np.dot(U, Y.T).T
    return np.sqrt(np.square(Ytr - X).sum()/X.shape[0])


def calc_RMSD(org, recovered_proteins):
    RMSDs = []
    X = np.array(org) # must be the original structure
    for protein in recovered_proteins:
        Y = np.array(protein)
        if X.shape == Y.shape:
            RMSDs.append(align_calc(X, Y))
    return RMSDs
        
def get_rmsd(algo, dataset, setting, lenght, org):
    Z = []
    for i in [5*x for x in range(1, lenght/5)]:
        row = []
        for j in [2*x for x in range(0, 11)]:
            trj = mdtraj.load_xyz(dataset+"\\"+algo+"\\"+setting+"\\n"+str(i)+"Noise"+str(j)+".xyz",top)
            row.append(np.mean(calc_RMSD(org, trj.xyz)))
        #endfor
        Z.append(row)  
        
    return Z
    


SAMPLE_COUNT = 10
data = pandas.read_csv("C:\Users\sepanta\Documents\University\CSIC5011\protein3D\PF00254_1R9H.csv", header=None)
data2 = pandas.read_csv("C:\Users\sepanta\Documents\University\CSIC5011\protein3D\PF00018_2HDA.csv", header=None)
data3 = pandas.read_csv("C:\Users\sepanta\Documents\University\CSIC5011\protein3D\PF00013_1WVN.csv", header=None)

pdb1 = get_Clean_Data(data)
pdb2 = get_Clean_Data(data2)
pdb3 = get_Clean_Data(data3)

do_experiments(pdb1, "PF00254_1R9H", SAMPLE_COUNT)
do_experiments(pdb2, "PF00018_2HDA", SAMPLE_COUNT)
do_experiments(pdb3, "PF00013_1WVN", SAMPLE_COUNT)

lengths = {"PF00254_1R9H":len(pdb1), "PF00018_2HDA":len(pdb2), "PF00013_1WVN":len(pdb3)}
datasets = {"PF00254_1R9H":pdb1, "PF00018_2HDA":pdb2, "PF00013_1WVN":pdb3}
noiseText = {"NormalNoise":"Normal Noise Standard Deviation", "UniformNoise": "Uniform Noise Max. Value"}


#plot the results
for dataset_name in datasets:
    for algo in ["Isomap", "LLE"]:
        for noise in ["NormalNoise", "UniformNoise"]:
            Z = get_rmsd(algo, dataset, noise, lengths[dataset], datasets[dataset_name])
    
            ylist = [5*x for x in range(1, lengths[dataset]/5)]
            xlist = [2*x for x in range(0, 11)]
            X, Y = np.meshgrid(xlist, ylist)
            plt.figure()
            cp = plt.contourf(X, Y, Z)
            plt.colorbar(cp, format=ticker.FuncFormatter(fmt))
            plt.title('RMDS Recovered by' + algo)
            plt.ylabel('No. Neighbours')
            plt.xlabel(noiseText[noise])
            fig1 = plt.gcf()
            plt.draw()
            fig1.savefig(+dataset +"_"+ noise +"_"+ algo, dpi=300)


