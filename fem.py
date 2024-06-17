import numpy as np
import seaborn as sns
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.spatial
import matplotlib.pyplot as plt
import meshio

# read the mesh from a file
mio = meshio.read('mesh/circle.inp')
p = mio.points.T[:2]
t = mio.cells[-1].data.T

# create affine mappings for all elements
A = p[:, t[[1, 2]]] - p[:, t[0]][:, None]
detA = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
invAT = 1 / detA * np.array([[A[1, 1], -A[1, 0]],
        [-A[0, 1], A[0, 0]]])

# find basis function gradients for all elements
dphihat = np.array([[-1, -1], [1, 0], [0, 1]])
dphi = np.einsum('ijk,lj', invAT, dphihat)

# assemble stiffness matrix K and load vector f
data, rows, cols = [], [], []
f = np.zeros(p.shape[1])
for i in range(3):
    v, dv = 1/3, dphi[:, :, i]
    for j in range(3):
        u, du = 1/3, dphi[:, :, j]
        data.append((du[0] * dv[0] + du[1] * dv[1]) * np.abs(detA) / 2)
        rows.append(t[i])
        cols.append(t[j])
    np.add.at(f, t[i], v * np.abs(detA) / 2)
K = sp.coo_matrix((np.hstack(data), (np.hstack(rows), np.hstack(cols)))).tocsr()

# solve linear system
u = np.zeros(K.shape[0])
# using 0.9999999 instead of 1 to avoid numerical approximation
I = np.nonzero((p[0]**2 + p[1]**2 < 0.9999999))[0]  
u[I] = sp.linalg.spsolve(K[I][:, I], f[I])

# plot solution
plt.tripcolor(*p, u, triangles=t.T, shading='gouraud')
plt.colorbar()
plt.axis('equal')
plt.title("Stress function")
plt.savefig('fem.png')
plt.close()