import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import random
from jax import config as c
c.update("jax_enable_x64", True)

from .util import tensor_product, tup, get_tensors

I = jnp.eye(2)
X = jnp.asarray([[0., 1.],[1., 0.]])
Y = jnp.asarray([[0., -1.j],[1.j, 0.]])
Z = jnp.asarray([[1., 0.],[0.,-1.]])
XX, YY, ZZ = jnp.kron(X,X), jnp.kron(Y,Y), jnp.kron(Z,Z)
XI, IX = jnp.kron(X,I), jnp.kron(I,X)
YI, IY = jnp.kron(Y,I), jnp.kron(I,Y)
ZI, IZ = jnp.kron(Z,I), jnp.kron(I,Z)
ZIZ = jnp.kron(jnp.kron(Z,I), Z)
SWAP = jnp.array([
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.]
])

mat = lambda v: v.reshape((4,4))

def operator_chain(N, ind, string_op, op):
    is_single_qubit_operator = True if len(op)==2 else False
    is_two_qubit_operator = True if len(op)==4 else False
    is_three_qubit_operator = True if len(op)==8 else False
    if is_single_qubit_operator: end=N-ind-1
    elif is_two_qubit_operator: end=N-ind-2
    elif is_three_qubit_operator: end=N-ind-3
    operators = [string_op] * ind + [op] + [string_op] * end
    return tensor_product(operators)

def get_nlayers(degree, n_repetitions):
    ''' Get the number of layers for a given TEBD circuit. '''
    
    if degree==1:
        n_SN_layers = 2*n_repetitions
    elif degree==2: 
        n_SN_layers = 2*n_repetitions+1
    elif degree==4:
        n_SN_layers = 10*n_repetitions+1

    return n_SN_layers

def get_brickwall_gate_matrix(t, pos='center', hamiltonian='ising-1d', **kwargs):
    '''
    pos: 'center', 'top', 'bottom', two_sites
    '''
    J = kwargs.get('J', 0.)
    g = kwargs.get('g', [0.,0.])
    h = kwargs.get('h', [0.,0.])

    if hamiltonian == 'ising-1d':
        if pos=='two_sites':
            return expm(-1j * t * (J * ZZ + g * XI + g * IX + h * ZI + h * IZ))
        g1, g2, h1, h2 = (g[0]/2, g[1]/2, h[0]/2, h[1]/2) if pos=='center' else (
            g[0] if pos=='top' else g[0]/2,
            g[1] if pos=='bottom' else g[1]/2,
            h[0] if pos=='top' else h[0]/2,
            h[1] if pos=='bottom' else h[0]/2
        )

        gate = expm(-1j * t * (J * ZZ + g1 * XI + g2 * IX + h1 * ZI + h2 * IZ))
        return gate
    
    elif hamiltonian=='heisenberg':
        h1, h2 = (h[0]/2, h[1]/2) if pos=='center' else (
            h[0] if pos=='top' else h[0]/2,
            h[1] if pos=='bottom' else h[1]/2
        )
        op1, op2 = [XX,YY,ZZ], [[XI,IX],[YI,IY],[ZI,IZ]]
        exp = jnp.zeros_like(XX)
        for i in range(3):
            exp += (J[i]*op1[i] + h1[i]*op2[i][0] + h2[i]*op2[i][1])
        gate = expm(-1j*t*exp)
        return gate    

def construct_ising_hamiltonian(N, J=1., g=0.75, h=0.6, disordered=False, get_matrix=False, key=None):
    if disordered: 
        sigmaJ, sigmag, sigmah = J/2, g/2, h/2
        keys = random.split(key, 3)
        _Js = random.uniform(keys[0], shape=(N-1,), minval=J-sigmaJ, maxval=J+sigmaJ)
        gs = random.uniform(keys[1], shape=(N,), minval=g-sigmag, maxval=g+sigmag)
        hs = random.uniform(keys[2], shape=(N,), minval=h-sigmah, maxval=h+sigmah)
    else:
        _Js, gs, hs = jnp.ones(N-1)*J, jnp.ones(N)*g, jnp.ones(N)*h

    # Prepare hamiltonian and Js matrix
    hamiltonian = jnp.zeros(2**N) if get_matrix else None
    Js = jnp.zeros((N,N))

    # Construct the Hamiltonian terms
    for j in range(N-1):
        Js = Js.at[j,j+1].set(_Js[j])
        if get_matrix:
            hamiltonian += operator_chain(N, j, I, _Js[j]*ZZ)
            hamiltonian += operator_chain(N, j, I, gs[j]*X) 
            hamiltonian += operator_chain(N, j, I, hs[j]*Z)
    if get_matrix: 
        hamiltonian += operator_chain(N, N-1, I, gs[N-1]*X) 
        hamiltonian += operator_chain(N, N-1, I, hs[N-1]*Z)
    
    return hamiltonian, Js, gs, hs

def construct_heisenberg_hamiltonian(N, J=[1,1,-.5], h=[.75,0,0], disordered=False, get_matrix=False, key=None):
    J, h = jnp.asarray(J), jnp.asarray(h)

    if disordered: 
        sigmaJ, sigmah = J/2, h/2
        keys = random.split(key, 2)
        _Js = random.uniform(keys[0], shape=(N-1,len(J)), minval=(J-sigmaJ), maxval=(J+sigmaJ))
        hs = random.uniform(keys[1], shape=(N,len(h)), minval=(h-sigmah), maxval=(h+sigmah))
    else:
        _Js, hs = jnp.broadcast_to(J, (N-1, len(J))), jnp.broadcast_to(h, (N, len(h)))
    hamiltonian = jnp.zeros(2**N) if get_matrix else None
    Js = jnp.zeros((N,N,3))
    sigma_sigma, sigma = [XX, YY, ZZ], [X, Y, Z]
    for j in range(N-1):
        for i in range(3):
            Js = Js.at[j,j+1,i].set(_Js[j,i])
            if get_matrix:
                hamiltonian += operator_chain(N, j, I, _Js[j,i]*sigma_sigma[i])
                hamiltonian += operator_chain(N, j, I, hs[j,i]*sigma[i])
    if get_matrix:
        for i in range(3):
            hamiltonian += operator_chain(N, N-1, I, hs[N-1,i]*sigma[i])
    return hamiltonian, Js, hs

def get_brickwall_trotter_gates_spin_chain(t, n_sites, n_repetitions=1, degree=2, hamiltonian='ising-1d', **kwargs):
    """
    Return the brickwall circuit gates for spin chains.
    Only implemented for even number of sites.

    """

    dt = t/n_repetitions
    J = kwargs.get('J', 0.)
    h = kwargs.get('h', 0.)
    g = kwargs.get('g', 0.)

    N_odd_gates, N_even_gates = int(n_sites/2), int(n_sites/2)  # Number of gates per layer
    if n_sites%2==0: N_even_gates -= 1
    odd_pairs = jnp.asarray([[i,i+1] for i in range(0,n_sites-1,2)])
    even_pairs = jnp.asarray([[i,i+1] for i in range(1,n_sites-1,2)])[::-1]  # Snaking order

    if degree in [1,2]:
        dt = dt/degree

        # Obtain the basic gates
        if hamiltonian=='ising-1d':
            gate_1 = get_brickwall_gate_matrix(dt, pos='top', hamiltonian=hamiltonian,
                                    J=J[tup(odd_pairs[0])], g=g[odd_pairs[0]], h=h[odd_pairs[0]])  # First edge gate
            gate_3 = get_brickwall_gate_matrix(dt, pos='bottom', hamiltonian=hamiltonian, 
                                    J=J[tup(odd_pairs[-1])], g=g[odd_pairs[-1]], h=h[odd_pairs[-1]])  # Last edge gate
            middle_gate = lambda _J, _g, _h: get_brickwall_gate_matrix(
                dt, pos='center', hamiltonian=hamiltonian, J=_J, g=_g, h=_h) 
        elif hamiltonian=='heisenberg':
            gate_1 = get_brickwall_gate_matrix(dt, pos='top', hamiltonian=hamiltonian, 
                                    J=J[tup(odd_pairs[0])], h=h[odd_pairs[0]])  # First edge gate
            gate_3 = get_brickwall_gate_matrix(dt, pos='bottom', hamiltonian=hamiltonian, 
                                    J=J[tup(odd_pairs[-1])], h=h[odd_pairs[-1]])  # Last edge gate
            middle_gate = lambda _J, _h: get_brickwall_gate_matrix(
                dt, pos='center', hamiltonian=hamiltonian, J=_J, h=_h) 
            
        # Obtain the basic layers
        L1, L2 = [gate_1.reshape((2,2,2,2))], []
        L1_qubits, L2_qubits = [tup(odd_pairs[0])], []
        L1_squared, L2_squared = [(gate_1@gate_1).reshape((2,2,2,2))], []
        L1_squared_qubits, L2_squared_qubits = [tup(odd_pairs[0])], []
        if hamiltonian=='ising-1d':
            for pair in odd_pairs[1:-1]:  # Odd layer
                gate = middle_gate(J[tup(pair)], g[pair], h[pair])
                L1.append(gate.reshape((2,2,2,2)))
                L1_squared.append((gate@gate).reshape((2,2,2,2)))
                L1_qubits.append(tup(pair))
                L1_squared_qubits.append(tup(pair))
            for pair in even_pairs:  # Even layer
                gate = middle_gate(J[tup(pair)], g[pair], h[pair])
                L2.append(gate.reshape((2,2,2,2)))
                L2_squared.append(((gate@gate).reshape((2,2,2,2))))
                L2_qubits.append(tup(pair))
                L2_squared_qubits.append(tup(pair))
        elif hamiltonian=='heisenberg':
            for pair in odd_pairs[1:-1]:  # Odd layer
                gate = middle_gate(J[tup(pair)], h[pair])
                L1.append(gate.reshape((2,2,2,2)))
                L1_squared.append((gate@gate).reshape((2,2,2,2)))
                L1_qubits.append(tup(pair))
                L1_squared_qubits.append(tup(pair))
            for pair in even_pairs:  # Even layer
                gate = middle_gate(J[tup(pair)], h[pair])
                L2.append(gate.reshape((2,2,2,2)))
                L2_squared.append((gate@gate).reshape((2,2,2,2)))
                L2_qubits.append(tup(pair))
                L2_squared_qubits.append(tup(pair))
        else: raise Exception('Hamiltonian not implemented')
        L1.append(gate_3.reshape((2,2,2,2)))
        L1_squared.append((gate_3@gate_3).reshape((2,2,2,2)))    
        L1_qubits.append(tup(odd_pairs[-1]))
        L1_squared_qubits.append(tup(odd_pairs[-1]))

        if degree==1:
            gates_per_step = [L1] + [L2]
            gates_per_step_qubits = [L1_qubits] + [L2_qubits]
            gates = gates_per_step*n_repetitions
            gates_qubits = gates_per_step_qubits*n_repetitions
        else:
            gates = [L1] + [L2_squared]
            gates_qubits = [L1_qubits] + [L2_squared_qubits]
            gates_per_step = [L1_squared] + [L2_squared]
            gates_per_step_qubits = [L1_squared_qubits] + [L2_squared_qubits]
            gates += gates_per_step*(n_repetitions-1)
            gates_qubits += gates_per_step_qubits*(n_repetitions-1)
            gates += [L1]
            gates_qubits += [L1_qubits]

        return gates, gates_qubits
    
    elif degree==4:
        s2 = (4-4**(1/3))**(-1)

        # V1 = U_2(s_2*t)
        V1, V1_qubits = get_brickwall_trotter_gates_spin_chain(2*s2*dt, n_sites, n_repetitions=2, degree=2, hamiltonian=hamiltonian, **kwargs)
        # V2 = U_2((1-4*s_2)*t)
        V2, V2_qubits = get_brickwall_trotter_gates_spin_chain((1-4*s2)*dt, n_sites, n_repetitions=1, degree=2, hamiltonian=hamiltonian, **kwargs)
        
        # Merge the last and first layers of V1, V2
        V11 = [(mat(v)@mat(v)).reshape((2,2,2,2)) for v in V1[0]]
        V11_qubits = [qubits for qubits in V1_qubits[0]]
        V12 = [(mat(v1)@mat(v2)).reshape((2,2,2,2)) for v1,v2 in zip(V1[-1],V2[0])]
        V12_qubits = [qubits for qubits in V1_qubits[-1]]
        V21 = [(mat(v2)@mat(v1)).reshape((2,2,2,2)) for v1,v2 in zip(V1[-1],V2[0])]
        V21_qubits = [qubits for qubits in V1_qubits[-1]]

        repeated_gates = V1[1:-1] + [V12] + V2[1:-1] + [V21] + V1[1:-1]
        repeated_gates_qubits = V1_qubits[1:-1] + [V12_qubits] + V2_qubits[1:-1] + [V21_qubits] + V1_qubits[1:-1]
        gates = [V1[0]] + repeated_gates
        gates_qubits = [V1_qubits[0]] + repeated_gates_qubits
        for _ in range(n_repetitions-1):
            gates += [V11]
            gates += repeated_gates
            gates_qubits += [V11_qubits]
            gates_qubits += repeated_gates_qubits
        gates += [V1[-1]]
        gates_qubits += [V1_qubits[-1]]

        return gates, gates_qubits

def get_brickwall_gate_matrix_TI(t, layer_is_odd=True, hamiltonian='ising-1d', **kwargs):
    J = kwargs.get('J', 0.)
    g = kwargs.get('g', 0.)
    h = kwargs.get('h', 0.)

    if hamiltonian == 'ising-1d':
        if layer_is_odd:
            gate = expm(-1j * t * (J * ZZ + g * XI + g * IX + h * ZI + h * IZ))
        else:
            gate = expm(-1j * t * J * ZZ)

    elif hamiltonian=='heisenberg':
        op1, op2 = [XX,YY,ZZ], [[XI,IX],[YI,IY],[ZI,IZ]]
        exp = jnp.zeros_like(XX)
        for i in range(3):
            exp += J[i]*op1[i]
        if layer_is_odd: 
            for i in range(3):
                exp += h[i]*(op2[i][0] + op2[i][1])
        gate = expm(-1j*t*exp)

    return gate

def get_brickwall_trotter_gates_spin_chain_TI(t, n_sites, n_repetitions=1, degree=2, hamiltonian='ising-1d', **kwargs):
    """
    Return the brickwall circuit gates for spin chains.
    Only implemented for even number of sites.

    """

    dt = t/n_repetitions

    n_layers = get_nlayers(degree, n_repetitions, n_id_layers=0, hamiltonian=hamiltonian)
    odd_pairs = jnp.asarray([[i,i+1] for i in range(0,n_sites-1,2)])
    even_pairs = jnp.asarray([[i,i+1] for i in range(1,n_sites-1,2)])[::-1]  # Snaking order
    gate_qubits = [odd_pairs if (ell % 2 == 0) else even_pairs for ell in range(n_layers)]


    if degree in [1,2]:
        # NOT FOR DEGREE=4
        dt = dt/degree

        odd_gate = get_brickwall_gate_matrix_TI(dt, layer_is_odd=True, hamiltonian=hamiltonian, **kwargs)
        even_gate = get_brickwall_gate_matrix_TI(dt, layer_is_odd=False, hamiltonian=hamiltonian, **kwargs)
        odd_gate_sq = (odd_gate@odd_gate).reshape((2,2,2,2))
        even_gate_sq = (even_gate@even_gate).reshape((2,2,2,2))
        odd_gate = odd_gate.reshape((2,2,2,2))
        even_gate = even_gate.reshape((2,2,2,2))

        if degree==1:
            gates = []
            for ell in range(n_layers):
                gates.append([odd_gate] if (ell % 2 == 0) else [even_gate])

        elif degree==2:
            gates = [[odd_gate]]
            for ell in range(1,n_layers-1):
                gates.append([odd_gate_sq] if (ell % 2 == 0) else [even_gate_sq])
            gates += [[odd_gate]]

        return gates, gate_qubits
    
    elif degree==4:
        s2 = (4-4**(1/3))**(-1)

        # V1 = U_2(s_2*t)
        V1, _ = get_brickwall_trotter_gates_spin_chain_TI(
            2*s2*dt, n_sites, n_repetitions=2, degree=2, hamiltonian=hamiltonian, **kwargs)
        # V2 = U_2((1-4*s_2)*t)
        V2, _ = get_brickwall_trotter_gates_spin_chain_TI(
            (1-4*s2)*dt, n_sites, n_repetitions=1, degree=2, hamiltonian=hamiltonian, **kwargs)
        
        # Merge the last and first layers of V1, V2
        V11 = [(mat(v)@mat(v)).reshape((2,2,2,2)) for v in V1[0]]
        V12 = [(mat(v1)@mat(v2)).reshape((2,2,2,2)) for v1,v2 in zip(V1[-1],V2[0])]
        V21 = [(mat(v2)@mat(v1)).reshape((2,2,2,2)) for v1,v2 in zip(V1[-1],V2[0])]

        repeated_gates = V1[1:-1] + [V12] + V2[1:-1] + [V21] + V1[1:-1]
        gates = [V1[0]] + repeated_gates
        for _ in range(n_repetitions-1):
            gates += [V11]
            gates += repeated_gates
        gates += [V1[-1]]

        return gates, gate_qubits