import numpy as np

def get_faces(lst):
    return [lst[:i] + lst[i+1:] for i in range(len(lst))]


def get_coeff(simplex, faces):
    if simplex in faces:
        idx = faces.index(simplex)
        return 1 if idx%2==0 else -1
    else:
        return 0

def boundary(complex):
    maxdim = len(max(complex, key=len))
    simplices = [sorted([spx for spx in complex if len(spx)==i]) for i in range(1,maxdim+1)]
    bnd = []
    for spx_k, spx_kp1 in zip(simplices, simplices[1:]):
        mtx = []
        for sigma in spx_kp1:
            faces = get_faces(sigma)
            mtx.append([get_coeff(spx, faces) for spx in spx_k])
        bnd.append(np.array(mtx).T)

    return bnd

def kernel(A, tol=1e-5):
    _, s, vh = np.linalg.svd(A)
    singular = np.zeros(vh.shape[0], dtype=float)
    singular[:s.size] = s
    null_space = np.compress(singular <= tol, vh, axis=0)
    return null_space.T

def cokernel(A, tol=1e-5):
    u, s, _ = np.linalg.svd(A)
    singular = np.zeros(u.shape[1], dtype=float)
    singular[:s.size] = s
    return np.compress(singular <= tol, u, axis=1)

def homology(boundary_ops, tol=1e-5):
    # Insert zero maps
    mm = boundary_ops[-1].shape[1]
    nn = boundary_ops[0].shape[0]
    boundary_ops.insert(0, np.ones(shape=(0, nn)))
    boundary_ops.append(np.ones(shape=(mm, 0)))

    H = []
    for del_k, del_kp1 in zip(boundary_ops, boundary_ops[1:]):
        kappa = kernel(del_k, tol)
        # Solve for psi
        psi, _, _, _ = np.linalg.lstsq(kappa, del_kp1, rcond=None)
        # Compute homology
        ksi = cokernel(psi, tol)
        H.append(np.dot(kappa, ksi))

    return H

def betti(H):
    return [basis.shape[1] for basis in H]


def main():
        bnd = boundary(creating_complex())
        print('boundary calculated')
        H = homology(bnd)
        print('homology calculated')
        b = betti(H)
        print(f'Betti numbers: {b}')


def creating_complex():
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R']

    for i in range(18):
      for j in range(i+1, 18):
        if j > (i + 1):
          letters.append(letters[i]+letters[j])
        if (j == (i + 1)) and (i % 2 == 1):
          letters.append(letters[i]+letters[j])

    stoplist = ['AFI', 'BEJ', 'LMR', 'KNQ', 'DEL', 'CKG', 'HIPQ', 'GJOR', 'DENQ', 'CFMR', 'ADIL', 'BCKJ', 'ADINQ', 'BCJMR', 'ADGNO', 'BCHMP', 'BEHPQ', 'AFGOR', 'GJKNO', 'HILMO', 'BEHLMP', 'AFGKNO']

    for i in range(18):
      for j in range(i+1, 18):
        for k in range(j+1, 18):
          comb = letters[i] + letters[j] + letters[k]
          if (comb not in stoplist) and ((comb[0] + comb[1]) not in stoplist) and ((comb[0] + comb[2]) not in stoplist) and ((comb[1] + comb[2]) not in stoplist):
            letters.append(comb)

    for i in range(18):
      for j in range(i+1, 18):
        for k in range(j+1, 18):
          for l in range(k+1, 18):
            comb = letters[i] + letters[j] + letters[k] + letters[l]
            if (comb not in stoplist) and ((comb[0] + comb[1]) not in stoplist) and ((comb[0] + comb[2]) not in stoplist) and ((comb[1] + comb[2]) not in stoplist) and ((comb[0] + comb[3]) not in stoplist) and ((comb[1] + comb[3]) not in stoplist) and ((comb[2] + comb[3]) not in stoplist):
              if ((comb[0] + comb[1] + comb[2]) not in stoplist) and ((comb[0] + comb[2] + comb[3]) not in stoplist) and ((comb[1] + comb[2] + comb[3]) not in stoplist):
                letters.append(comb)

    for i in range(18):
      for j in range(i+1, 18):
        for k in range(j+1, 18):
          for l in range(k+1, 18):
            for m in range(l+1, 18):
              comb = letters[i] + letters[j] + letters[k] + letters[l] +letters[m]
              if (comb not in stoplist) and ((comb[0] + comb[1]) not in stoplist) and ((comb[0] + comb[2]) not in stoplist) and ((comb[1] + comb[2]) not in stoplist) and ((comb[0] + comb[4]) not in stoplist) and ((comb[1] + comb[4]) not in stoplist) and ((comb[2] + comb[4]) not in stoplist) and ((comb[3] + comb[4]) not in stoplist):
                if ((comb[0] + comb[1] + comb[2]) not in stoplist) and ((comb[0] + comb[2] + comb[3]) not in stoplist) and ((comb[1] + comb[2] + comb[3]) not in stoplist) and ((comb[0] + comb[1] + comb[4]) not in stoplist) and ((comb[0] + comb[2] + comb[4]) not in stoplist) and ((comb[0] + comb[3] + comb[4]) not in stoplist) and ((comb[1] + comb[2] + comb[4]) not in stoplist) and ((comb[1] + comb[3] + comb[4]) not in stoplist) and ((comb[2] + comb[3] + comb[4]) not in stoplist):
                  if ((comb[1] + comb[2] + comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[2] + comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[1] + comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[1] + comb[2] + comb[4]) not in stoplist) and ((comb[0] + comb[1] + comb[2] + comb[3]) not in stoplist):
                    letters.append(comb)

    for i in range(18):
      for j in range(i+1, 18):
        for k in range(j+1, 18):
          for l in range(k+1, 18):
            for m in range(l+1, 18):
              for n in range(m+1, 18):
                comb = letters[i] + letters[j] + letters[k] + letters[l] + letters[m] + letters[n]
                if (comb not in stoplist) and ((comb[0] + comb[1]) not in stoplist) and ((comb[0] + comb[2]) not in stoplist) and ((comb[1] + comb[2]) not in stoplist) and ((comb[0] + comb[4]) not in stoplist) and ((comb[1] + comb[4]) not in stoplist) and ((comb[2] + comb[4]) not in stoplist) and ((comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[5]) not in stoplist) and ((comb[1] + comb[5]) not in stoplist) and ((comb[2] + comb[5]) not in stoplist) and ((comb[3] + comb[5]) not in stoplist) and ((comb[4] + comb[5]) not in stoplist):
                  if ((comb[0] + comb[1] + comb[2]) not in stoplist) and ((comb[0] + comb[2] + comb[3]) not in stoplist) and ((comb[1] + comb[2] + comb[3]) not in stoplist) and ((comb[0] + comb[1] + comb[4]) not in stoplist) and ((comb[0] + comb[2] + comb[4]) not in stoplist) and ((comb[0] + comb[3] + comb[4]) not in stoplist) and ((comb[1] + comb[2] + comb[4]) not in stoplist) and ((comb[1] + comb[3] + comb[4]) not in stoplist) and ((comb[2] + comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[1] + comb[5]) not in stoplist) and ((comb[0] + comb[2] + comb[5]) not in stoplist) and ((comb[0] + comb[3] + comb[5]) not in stoplist) and ((comb[0] + comb[4] + comb[5]) not in stoplist) and ((comb[1] + comb[2] + comb[5]) not in stoplist) and ((comb[1] + comb[3] + comb[5]) not in stoplist) and ((comb[1] + comb[4] + comb[5]) not in stoplist) and ((comb[2] + comb[3] + comb[5]) not in stoplist):
                    if ((comb[1] + comb[2] + comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[2] + comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[1] + comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[1] + comb[2] + comb[4]) not in stoplist) and ((comb[0] + comb[1] + comb[2] + comb[3]) not in stoplist) and ((comb[0] + comb[1] + comb[2] + comb[5]) not in stoplist) and ((comb[0] + comb[1] + comb[3] + comb[5]) not in stoplist) and ((comb[0] + comb[1] + comb[4] + comb[5]) not in stoplist) and ((comb[0] + comb[2] + comb[3] + comb[5]) not in stoplist) and ((comb[0] + comb[2] + comb[4] + comb[5]) not in stoplist) and ((comb[0] + comb[3] + comb[4] + comb[5]) not in stoplist) and ((comb[1] + comb[2] + comb[3] + comb[5]) not in stoplist) and ((comb[1] + comb[2] + comb[4] + comb[5]) not in stoplist) and ((comb[1] + comb[3] + comb[4] + comb[5]) not in stoplist) and ((comb[2] + comb[3] + comb[4] + comb[5]) not in stoplist):
                      if ((comb[0] + comb[1] + comb[2] + comb[3] + comb[4]) not in stoplist) and ((comb[0] + comb[1] + comb[2] + comb[3] + comb[5]) not in stoplist) and ((comb[0] + comb[1] + comb[2] + comb[4] + comb[5]) not in stoplist) and ((comb[0] + comb[1] + comb[3] + comb[4] + comb[5]) not in stoplist) and ((comb[0] + comb[2] + comb[3] + comb[4] + comb[5]) not in stoplist) and ((comb[1] + comb[2] + comb[3] + comb[4] + comb[5]) not in stoplist):
                        letters.append(comb)


    return letters



main()