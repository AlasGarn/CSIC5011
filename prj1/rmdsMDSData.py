import mdtraj
import zipfile
import numpy as np
from scipy.linalg import svd
import pandas as pd
import matplotlib.pyplot as plt


MAX_PARTS = 2
MAX_MISSING = 27
MAX_NOISE = 20

def get_Clean_Data(data):
    cleanData = []
    for index, row in data.iterrows():
        cleanData.append(row.values[0].split()[6:])
        
    cleanData= [map(float, row) for row in cleanData]  
    return cleanData


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

def at_fmt(resi, xyz, resn):
    st = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}\n"\
        .format("ATOM", 0, "C"," ",str(resn), "A", resi, " ", *xyz)
    return st

# write trajectory
def write_traj(pdb_iso, file_name):
    protein = {"ns":[],"xyzs":[]}
    for index, row in pdb_iso.iterrows():
        protein["ns"].append(row[0])
        protein["xyzs"].append(map(float,row[1:]))
    with open(file_name,"w") as f:
        f.write(str(len(protein["ns"])) + "\n\n")
        xyz = [" ".join(map(str, [np.round(x,3) for x in xyz])) for xyz in protein["xyzs"]]
        ns =  map(str, protein["ns"])
        lines = ["C" + " " + b for a, b in zip(ns, xyz)]
        f.write("\n".join(lines) + "\n")
    return

for i in range(1, MAX_MISSING+1):
    for j in range(1, MAX_NOISE+1):
        for part in range(1, MAX_PARTS+1):
            content = []
            dt = pd.read_csv("zipfile\\data1 unif max "+str(j)+" missing "+str(i)+" orignal"+str(part)+".csv")
            proteins = []
            protein = {"ns":[],"names":[],"xyzs":[]}
            
            for index, row in dt.iterrows():
                n = row[0]
                name = row[0]
                x = row[1]
                y = row[2]
                z = row[3]
                protein["ns"].append(int(n))
                protein["names"].append(int(n))
                protein["xyzs"].append(map(float,[x, y, z]))
            proteins.append(protein)
            
            # write topology
            top = proteins[0]
            lines = []
            with open("zipfile\\data1 unif max "+str(j)+" missing "+str(i)+" orignal"+str(part)+".pdb","w") as f:
                for ln in zip(*top.values()):
                    lines.append(at_fmt(*ln))
                f.writelines(lines)

for i in range(1, MAX_MISSING+1):
    for j in range(1, MAX_NOISE+1):
        for part in range(1, MAX_PARTS+1):
            dt = pd.read_csv("zipfile\\data1 unif max "+str(j)+" missing "+str(i)+" part"+str(part)+".csv")
            write_traj(dt, "zipfile\\data1 unif max "+str(j)+" missing "+str(i)+" part"+str(part)+".xyz")
        
Z = []
for i in range(1, MAX_MISSING+1):
    row = []
    for j in range(1, MAX_NOISE+1):
        val = []
        for part in range(1, MAX_PARTS+1):
            data = pd.read_csv("zipfile\\data1 unif max "+str(j)+" missing "+str(i)+" orignal"+str(part)+".pdb", header=None)
            top = mdtraj.load("zipfile\\data1 unif max "+str(j)+" missing "+str(i)+" orignal"+str(part)+".pdb").topology
            org_structure = get_Clean_Data(data)
            trj = mdtraj.load_xyz("zipfile\\data1 unif max "+str(j)+" missing "+str(i)+" part"+str(part)+".xyz",top)
            val.append(calc_RMSD(org_structure, trj.xyz)[0])
        row.append(np.mean(val))
    Z.append(row)

xlist = range(1, MAX_NOISE+1)
ylist = range(1, MAX_MISSING +1)
X, Y = np.meshgrid(xlist, ylist)
plt.figure()
cp = plt.contourf(X, Y, Z)
plt.colorbar(cp)
plt.title('RMSD of the Structures Recovered by MDS')
plt.ylabel('No. Points Missing')
plt.xlabel("Noise Standard Deviation")
fig1 = plt.gcf()
plt.draw()
fig1.savefig("MDS_RMSD", dpi=300)


