
def morseBoundIndex(multiplicities, index):
    # Computes the right-hand side of Morse's inequality given a dictionary of
    # multiplicities and the upper index for the sum.
    # Precondition: 0 <= index <= dim M.
    return sum([(-1)^i * multiplicities[i] for i in range(0,index + 1) if i in multiplicities])


def gramSchmidt(B):
    if len(B) == 0 or len(B[0]) == 0:
        return B, matrix(RR, 0, 0, [])
    n = len(B)
    Bstar = [B[0]]
    zero = vector(RR, len(B[0]))
    if Bstar[0] == zero:
        raise ValueError("linearly dependent input for module version of Gram-Schmidt")
    mu = matrix(RR, n, n)
    for i in range(1, n):
        for j in range(i):
            mu[i, j] = B[i].dot_product(Bstar[j]) / (Bstar[j].dot_product(Bstar[j]))
        Bstar.append(B[i] - sum(mu[i, j] * Bstar[j] for j in range(i)))
        if Bstar[i] == zero:
            raise ValueError("linearly dependent input for module version of Gram-Schmidt")
    return matrix(RR,Bstar), matrix(RR,mu)

def morseBounds(g, v, variables):
    # Morse function:
    f = v.dot_product(vector(variables))
    n = len(variables)
    # Define Lagrange function:
    l = var('l')
    L = f - l * g

    # Find critical points:
    solutions = solve([diff(L,variable) == 0 for variable in variables] + [diff(L,l) == 0], variables + [l])

    print("Critical points of f : ", solutions, "\n")

    # Compute an orthonormal basis extending v. This basis contains a normalised version of v:
    basisFromV = [v] + VectorSpace(RR, n).subspace([v]).complement().basis()
    orthogonalBasis, mu = gramSchmidt(basisFromV)
    orthonormalBasis =  (orthogonalBasis*orthogonalBasis.transpose()).apply_map(sqrt).inverse() * orthogonalBasis

    # Compute the gradient of g:
    gradG = vector([g.derivative(variable) for variable in variables])

    # Check if 0 is regular value of g:
    solve([gradG == 0, g == 0], variables)

    # Compute the directional derivative dg/dv (x):
    vg = gradG.dot_product(v)

    # Compute grad(dg/dv(x)):
    dvg = vector([vg.derivative(variable) for variable in variables])

    # Compute our magic bilinear form Hx (as a matrix):
    Hx = (1/vg * (1/vg * gradG.outer_product(dvg) - g.hessian()))

    # Compute the action of Hx on the orthogonal complement of v
    HxSubspace = (orthonormalBasis * Hx * orthonormalBasis.transpose())[1:,1:]

    # Compute the index of f at each critical point:
    indices = []

    for sol in solutions:
        eigenvalues = HxSubspace.subs(sol).eigenvalues()
        index = sum(1 for eigenvalue in eigenvalues if eigenvalue < 0)
        indices.append(index)

    # Gather the number of critical points per index:
    multiplicities = {}

    # Compute the frequency of each element:
    for index in indices:
        if index in multiplicities:
            multiplicities[index] += 1
        else:
            multiplicities[index] = 1

    # Display the frequency dictionary:
    print("Critical points of f : ", solutions, "\n")
    print("Indices at these points : ", indices, "\n")
    print("Multiplicities of indices : ", multiplicities, "\n") 
    print("Strong Morse inequalities : \n", " ".join([f"sum_[i = 0...{id}] (-1)^i b_i <= {morseBoundIndex(multiplicities, id)}," for id in range(0,n)]), "\n")
    print("Weak Morse inequalities : \n", " ".join([f"b_{id} <= {multiplicities[id] if id in multiplicities else 0}," for id in range(0,n)]), "\n")

# Example usage: compute Morse indices of the 5-sphere:
variables = [variable for variable in var(' '.join([f"x{i}" for i in range(3)]))]
v = vector([1,0,0])

# Constraint g, g(u,v,w,x,y,z) = 0 on the 2-sphere:
g = x

morseBounds(g,v,variables)
