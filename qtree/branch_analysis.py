import sys
import qutip as qt
import matplotlib.pyplot as plt
import scqubits as scq
import numpy as np
from tqdm import tqdm
from scipy.linalg import eigh  
from scipy.sparse.linalg import eigsh  
from scipy.linalg import cosm
from dataclasses import dataclass
from typing import Optional
from .coupled_fluxonium import CoupledFluxonium
from .coupled_transmon import CoupledTransmon

# TBD: Rename PCA to something more accurate (overlap matrix)

"""""""""
Pt. 0: Gathering The Troops
"""""""""

# Class structure for passing around lots of params
@dataclass
class ObjPackage:
    rho_set      : Optional[np.ndarray] = None
    H_coupled_mat: Optional[np.ndarray] = None
    c_evals      : Optional[np.ndarray] = None
    c_evecs      : Optional[np.ndarray] = None
    H_qubit_mat  : Optional[np.ndarray] = None
    q_evals      : Optional[np.ndarray] = None
    q_evecs      : Optional[np.ndarray] = None
    H_res_mat    : Optional[np.ndarray] = None
    r_evals      : Optional[np.ndarray] = None
    r_evecs      : Optional[np.ndarray] = None
    H_cm_mat     : Optional[np.ndarray] = None
    cm_evals     : Optional[np.ndarray] = None
    cm_evecs     : Optional[np.ndarray] = None
    qdim         : Optional[int] = None
    rdim         : Optional[int] = None
    cdim         : Optional[int] = None
    get_full     : Optional[bool] = False

def oscillator_hamiltonian(omega_r, dim):
    """
    Make oscillator hamiltonian in the number basis
    
    Inputs:
      - omega_r   : resonator frequency in GHz                     (float)
      - dim       : truncated dimension of resonator hilbert space (int)
      
    Returns:
      - H         : resonator hamiltonian (numpy array)
      - a         : annihilation operator (numpy array)
      - adag      : creation operator     (numpy array)
    """
    # Create annihilation operator
    a = np.zeros((dim, dim), dtype=complex)
    
    for n in range(1, dim):
        a[n-1, n] = np.sqrt(n)
        
    adag = a.conj().T
    num  = adag @ a
    H    = omega_r * (num)
    return H, a, adag

def fast_ptrace_qubit(c_evecs, qdim, rdim, cdim=None):
    """
    Compute the reduced density matrices for the qubit subspace, tracing over
    the resonator degrees of freedom and optionally the chain mode degrees of 
    freedom.
    
    Inputs:
      - c_evecs           : dressed eigenvectors of full hamiltonian     (numpy array)
      - qdim              : qubit subspace dimension                     (int)
      - rdim              : resonator subspace dimension                 (int)
      - cdim (optional)   : chain mode subspace dimension                (int)
      
    Returns:
      - rho               : reduced density matrix in qubit subspace (numpy array)
    """
    if cdim is not None:
        # Find number of eigenstates
        num_eigs = c_evecs.shape[0] # shape (num_eigs, qdim*rdim)
        # Reshape to (num_eigs, qdim, rdim)
        psi = c_evecs.reshape(num_eigs, qdim, rdim, cdim)
        # Compute rho = psi @ psi^† over resonator index
        rho = np.einsum('nqrc,nprc->npq', psi, psi.conj()) # shape (num_eigs, qdim, qdim)
    else:
        # Find number of eigenstates
        num_eigs = c_evecs.shape[0] # shape (num_eigs, qdim*rdim)
        # Reshape to (num_eigs, qdim, rdim)
        psi = c_evecs.reshape(num_eigs, qdim, rdim)
        # Compute rho = psi @ psi^† over resonator index
        rho = np.einsum('nqr,npr->npq', psi, psi.conj()) # shape (num_eigs, qdim, qdim)

    return rho

def get_objs(params, qubit_type, H_full=None, update_flux=False):
    """
    Gather all subspace eigenvectors/values, density matrices and hamiltonians.
    
    Inputs:
      - params             : simulation parameters                                       (dict)
      - qubit_type         : name of qubit architecture                                  (string)
      - H_full             : full Hilbert space                                          (custom class object)
      - update_flux        : conditionally updates flux in hamiltonian before operations (bool)
      
    Returns:
      - pkg                : all inputs necessary for branch_analysis         (DataClass package)
      - H_full             : returns full Hilbert space, may not be necessary (custom class object)
    """

    pkg = ObjPackage()
    # Gather qubit/hamiltonian objects
    if (not update_flux):
        # Check device type, get full Hilbert space/qubit subspace
        if qubit_type == 'fluxonium':
            H_full = CoupledFluxonium(params)
            qubit = H_full.fluxonium
        if qubit_type == 'transmon':
            H_full = CoupledTransmon(params)
            qubit = H_full.transmon
            
    # Update flux if needed
    else:
        flux = params['flux'][0]
        H_full.update_flux(flux)
        if qubit_type == 'fluxonium':
            qubit = H_full.fluxonium
        if qubit_type == 'transmon':
            qubit = H_full.transmon

    # Get hilbert space data
    hs = H_full.hs
    # Get resonator subspace/dimension
    resonator = H_full.resonator
    qdim      = qubit.truncated_dim
    rdim      = resonator.truncated_dim
    
    # Get all the eigenvalues/vectors/density matrices
    H_coupled_mat      = hs.hamiltonian().full()
    H_qubit_mat        = qubit.hamiltonian()
    H_res_mat, a, adag = oscillator_hamiltonian(H_full.f_r, rdim)
    q_evals, q_evecs   = qubit.eigensys(qdim)
    r_evals, r_evecs   = resonator.eigensys(rdim)

    # If chain mode present, gather it's objects and package them
    if len(H_full.hs.subsystem_list) == 3:
        cdim                    = H_full.hs.subsystem_list[2].truncated_dim
        chain_mode              = H_full.chain_mode
        cm_evals, cm_evecs      = chain_mode.eigensys(cdim)
        H_cm_mat, a_cm, adag_cm = oscillator_hamiltonian(H_full.f_c, cdim)

        c_evals, qc_evecs = hs.eigensys(qdim*rdim*cdim)
        c_evecs           = np.asarray([vec.full().T[0] for vec in qc_evecs])
        rho_set           = fast_ptrace_qubit(c_evecs, qdim, rdim, cdim)
        
        # packing package (chain mode)
        pkg.H_cm_mat = H_cm_mat
        pkg.cm_evals = cm_evals
        pkg.cm_evecs = cm_evecs
        pkg.cdim     = cdim

    else:
        c_evals, qc_evecs = hs.eigensys(qdim*rdim)    
        c_evecs = np.asarray([vec.full().T[0] for vec in qc_evecs])
        rho_set = fast_ptrace_qubit(c_evecs, qdim, rdim)
    
    # packing package
    pkg.rho_set       = rho_set
    pkg.H_coupled_mat = H_coupled_mat
    pkg.c_evals       = c_evals
    pkg.c_evecs       = c_evecs
    pkg.H_qubit_mat   = H_qubit_mat
    pkg.q_evals       = q_evals
    pkg.q_evecs       = q_evecs
    pkg.H_res_mat     = H_res_mat
    pkg.r_evals       = r_evals
    pkg.r_evecs       = r_evecs
    pkg.qdim          = qdim
    pkg.rdim          = rdim
    return pkg, H_full

def ladder_levels(vecs, c_evecs, b_adag, new_vecs=False):
    """
    Vectorized calculation of all overlaps between bare/dressed states for
    a given collection of bare states (ideally within one <n_r> = c branch). 
    Can optionally apply ladder operators to climb up <n_r> branches.
    
    Inputs:
      - vecs       : collection of "bare" states within one branch        (numpy array)
      - c_evecs    : collection of all dressed states                     (numpy array)
      - b_adag     : ladder operator for resonator subspace               (numpy array)
      - new_vecs   : conditionally applies ladder operator to bare states (bool)
      
    Returns:
      - PCA        : matrix of all overlaps for determining index assignment (numpy array)
    """
    
    if new_vecs:
        vecs = [b_adag @ vec for vec in vecs]  # update vecs

    # Convert lists of Qobj to NumPy arrays
    vecs_array    = np.column_stack([vec.squeeze() for vec in vecs])     # (N_total, N_vecs)
    c_evecs_array = np.column_stack([vec.squeeze() for vec in c_evecs])  # (N_total, N_cvecs)

    # Compute PCA in fully vectorized way
    PCA = np.abs(vecs_array.conj().T @ c_evecs_array)**2  # (len(vecs), len(cvecs))

    return PCA


def get_map(PCA, used_indices=None):
    """
    Given a matrix containing all overlaps, check and assign indices as best as 
    possible, ensuring no duplicates.
    
    Inputs:
      - PCA             : matrix of bare state/dressed state overlaps          (numpy array)
      - unsed_indices   : set of indices which have been used in previous maps (set)

    Returns:
      - final_map       : list of indices mapping bare states to dressed states 1:1    (list)
      - assigned        : updated set of indices which have been used in previous maps (set)

    """
    if used_indices is None:
        used_indices = set()

    mapp = np.argmax(PCA, axis=0)
    assigned = set()  # indices assigned in this call
    final_map = []

    for i, dressed_idx in enumerate(mapp):
        # Candidate must not be in either assigned (this call) or used_indices (previous calls)
        if dressed_idx not in assigned and dressed_idx not in used_indices:
            final_map.append(dressed_idx)
            assigned.add(dressed_idx)
        else:
            # Pick next best dressed state not used yet
            column = PCA[:, i]
            candidates = np.argsort(column)[::-1]  # descending
            for c in candidates:
                if c not in assigned and c not in used_indices:
                    final_map.append(c)
                    assigned.add(c)
                    break
            else:
                # Fallback: if somehow all indices are exhausted, just pick the next free
                remaining = set(range(PCA.shape[0])) - assigned - used_indices
                if remaining:
                    c = remaining.pop()
                    final_map.append(c)
                    assigned.add(c)
                else:
                    raise RuntimeError("No available indices left to assign!")
    # Return both final_map and newly assigned indices
    return final_map, assigned

def branch_analysis(pkg: ObjPackage, update_flux, ncrit_mode=False):
    """
    Inputs:
      - H_coupled_mat        : full coupled hamiltonian (N,N), N = qdim*rdim*cdim          (numpy array)
      - H_qubit_mat          : qubit subspace hamiltonian (qdim * qdim)                    (numpy array)
      - q_evals, q_evecs     : bare qubit eigenvectors/eigenvalues                         (1D numpy array, 2D numpy array)
      - H_res_mat            : resonator subspace hamiltonian (rdim * rdim)                (numpy array)
      - r_evals, r_evecs     : bare resonator eigenvectors/eigenvalues                     (1D numpy array, 2D numpy array)
      - qdim, rdim           : dimension of qubit/resonator subspaces                      (int, int)
      - update_flux          : conditionally updates flux in hamiltonian before operations (bool)
      
      ______________________
      ~------Optional------~
      ______________________
      
      - H_cm_mat             : chain mode hamiltonian (cdim * cdim)    (2D numpy array)
      - cm_evals, cm_evecs   : chain mode eigenvalues and eigenvectors (1D numpy array, 2D numpy array)
      - cdim                 : chain mode truncated dimension          (int)
      
      
    Returns:
      - params               : contains parameters used in simulation    (dict)
      - data                 : contains all data produced by simulation  (dict)
      - params_list          : conveniently repacked params for plotting (list)
      - data_list            : conveniently repacked data for plotting   (list)
      - branches             : dictionary containing all branch info     (dict)
    """

    # unpacking package
    rho_set        = pkg.rho_set
    H_coupled_mat  = pkg.H_coupled_mat
    c_evecs        = pkg.c_evecs
    H_qubit_mat    = pkg.H_qubit_mat
    q_evals        = pkg.q_evals
    q_evecs        = pkg.q_evecs
    H_res_mat      = pkg.H_res_mat
    r_evals        = pkg.r_evals
    r_evecs        = pkg.r_evecs
    H_cm_mat       = pkg.H_cm_mat
    cm_evals       = pkg.cm_evals
    cm_evecs       = pkg.cm_evecs
    qdim           = pkg.qdim
    rdim           = pkg.rdim
    cdim           = pkg.cdim
    get_full       = pkg.get_full
    
    """""""""
    Pt. 1: Object Initialization
    """""""""
    # Obtain dressed eigenvalues/eigenvectors
    dressed_evals, dressed_evecs = eigh(H_coupled_mat)
    # Reorganizing vals/vecs
    qvecs_q = [qt.Qobj(q_evecs[:, -j], dims=[[qdim],[1]]) for j in range(qdim)]

    # Create empty array to form ladder operator
    a = np.zeros((rdim, rdim), dtype=complex)
    
    # Populate ladder operator values
    for n in range(1, rdim):
        a[n-1, n] = np.sqrt(n)
                           
    # Create a^{\dagger}
    adag = a.conj().T

    # Create number operator
    n_res_op = adag @ a

    # Create qubit population operator
    big_n_q  = np.diag(np.arange(qdim)).astype(complex)
    
    # Convenient renaming
    V = dressed_evecs

    # Important to remember shapes here
    # rho_set: (N_states, qdim, qdim)
    # big_n_q: (qdim, qdim)
    # n_q_expect: (N_states,)
    n_q_expect = np.einsum('nij,ij->n', rho_set, big_n_q).real
    
    # Create resonator number operator same dimension as full H
    if cdim:
        # Create empty array to form ladder operator
        c_a = np.zeros((cdim, cdim), dtype=complex)
        
        # Populate ladder operator values
        for n in range(1, cdim):
            c_a[n-1, n] = np.sqrt(n)
                               
        # Create a^{\dagger}
        c_adag = c_a.conj().T
    
        # Create number operator
        n_chain_op = c_adag @ c_a
        
        b_adag  = np.kron(np.eye(qdim, dtype=complex), np.kron(adag, np.eye(cdim, dtype=complex)))
        big_n   = np.kron(np.eye(qdim, dtype=complex), np.kron(n_res_op, np.eye(cdim, dtype=complex)))
        big_n_c = np.kron(np.eye(qdim, dtype=complex), np.kron(np.eye(rdim, dtype=complex), n_chain_op))

    else:
        b_adag = np.kron(np.eye(qdim, dtype=complex), adag)
        big_n  = np.kron(np.eye(qdim, dtype=complex), n_res_op)

    """""""""
    Pt. 2: Expectation Values
    """""""""
    
    # Evil Einstein sum for vectorized expectation values
    if cdim:
        big_n_c_V  = big_n_c @ V                                     # (N, M)
        n_c_expect = np.einsum('ij,ij->j', V.conj(), big_n_c_V).real # (M, )
        #M = V.reshape((qdim, rdim, cdim, V.shape[1]), order='C')     # shape (qdim, rdim, cdim, M)
    #else:
    big_n_V    = big_n @ V                                       # (N, M)
    n_r_expect = np.einsum('ij,ij->j', V.conj(), big_n_V).real   # (M,)
    #M = V.reshape((qdim, rdim, V.shape[1]), order='C')           # shape (qdim, rdim, M)
        
    # Build rotation matrix from original bare qubit eigenvectors
    # Here, q_evecs is (qdim, qdim)
    U = np.column_stack([vec for vec in q_evecs])
    q_evecs = np.asarray([(U.conj().T @ vec) for vec in q_evecs])

    """""""""
    Pt. 3: Initial Branch Assignment
    """""""""
    
    # Get resonator vacuum state
    r_vac = r_evecs[0]
    # Construct qubit x vacuum states for first round of mapping
    #for n_c in range(cdim)
    if cdim:
        bare_state_big = []
        for n_c in range(cdim):
            tot_vac = np.kron(r_vac, cm_evecs[n_c])  # Explicitly handle absence of chain mode
            bare_state_big.append(np.kron(q_evecs, tot_vac))  # kron with qubit eigenvectors
        bare_state_big = np.asarray(bare_state_big)
        bare_states = bare_state_big.reshape(-1, bare_state_big.shape[2]) # I cannot believe that this is fine
        PCA = ladder_levels(bare_states, c_evecs, b_adag)

    else:
        bare_states = np.kron(q_evecs, r_vac) # (qdim*rdim, qdim)
        PCA = ladder_levels(bare_states, c_evecs, b_adag)
        
    PCA_full = PCA.T
    # Obtain assignments and set of assigned indices
    fixed_map, assigned = get_map(PCA_full)
    # Initialize assignment objects
    PCA_list = [PCA_full]
    map_list = [fixed_map]
    used_idx = set(assigned)
    
    """""""""
    Pt. 4: Iterative Branch Assignment
    """""""""
    
    # Apply ladder operators and re-assign map up to max n_r
    for i in tqdm(range(rdim-1), ascii=True, desc="climbing ladders...", disable=update_flux):
        vecs = c_evecs[np.asarray(map_list[-1])]
        PCA_temp = ladder_levels(vecs, c_evecs, b_adag, new_vecs=True)
        PCA_list.append(PCA_temp.T)
        fixed_map_temp, assigned = get_map(PCA_temp.T, used_indices=used_idx)
        used_idx = used_idx | assigned
        map_list.append(fixed_map_temp)
        
    """""""""
    Pt. 5: Packaging
    """""""""
    
    # Build lists n_list1 (resonator excitation), 
    # n_list2 (qubit excitation) in dressed order
    n_list1 = n_r_expect.tolist()
    n_list2 = n_q_expect.tolist()
    if cdim:
        n_list3 = n_c_expect.tolist()  # Chain excitation for labeling
    
    # Bin branches based on map list
    branches = {}
    for ti in range(len(map_list[0])):   # Number of bare qubit states
        idx_list = [map_list[nj][ti] for nj in range(len(map_list))]
        n_r_arr  = np.abs(np.array(n_list1)[idx_list])
        n_q_arr  = np.abs(np.array(n_list2)[idx_list])
        if cdim:
            n_c_arr = np.abs(np.array(n_list3)[idx_list])
            key = 'q=' + str(np.round(n_q_arr[0])) + 'c=' + str(n_c_arr[0])
            branches[key] = ((n_r_arr, n_q_arr, n_c_arr), None, None)
        else:
            branches[np.round(n_q_arr[0])] = ((n_r_arr, n_q_arr), None, None)
            
    # Params, data packaging
    params = {"qdim": (qdim, None, None), 
              "rdim": (rdim, None, None), 
              "cdim": (cdim, None, None)}
    
    data = {
        "PCA_list"   : (PCA_list, None, None),
        "eigenvalues": (dressed_evals, 'GHz', None),
        "map_list"   : (map_list, None, None),
        "avg_n_r"    : (n_list1, None, None),
        "avg_n_q"    : (n_list2, None, ["avg_n_r"])}
    if cdim:
        data["avg_n_c"] = (n_list3, None, ["avg_n_r"])

    # Build lists for plotting
    data_list = []
    params_list = [params for _ in range(len(dressed_evals))]
    labels = []
    
    # For each bare state
    for t_idx in range(np.asarray(map_list).shape[1]):
        idx_list = []
        data_temp = data.copy()  # start from global data dict
        for nj in range(np.asarray(map_list).shape[0]):
            psi_index = map_list[nj][t_idx]
            idx_list.append(psi_index)

        # Extract expectation values along this branch
        n_r_branch = np.abs(np.asarray(n_list1)[idx_list])
        n_q_branch = np.abs(np.asarray(n_list2)[idx_list])

        # Sort by resonator population
        sort_idx   = np.argsort(n_r_branch)
        n_r_branch = n_r_branch[sort_idx]
        n_q_branch = n_q_branch[sort_idx]
        
        if cdim:
            n_c_branch = np.abs(np.asarray(n_list3)[idx_list])
            n_c_branch = n_c_branch[sort_idx]

            # Label this branch by initial qubit population
            q_idx = t_idx % qdim
            c_idx = t_idx // qdim
            label = f"q{q_idx}_c{c_idx}"
        else:
            label = str(np.round(n_q_branch[0]))

        labels.append(label)

        if get_full:
            # Populate data_temp
            data_temp['n_r'] = (n_r_branch, None, None)
            data_temp['n_q'] = (n_q_branch, None, ['n_r'])
            if cdim:
                data_temp['n_c'] = (n_c_branch, None, ['n_r'])
            # Append to data_list
            data_list.append(data_temp)
            # Add to global data dictionary, label is qubit pop
            data[f'n_r_branch={label}'] = (n_r_branch, None, None)
            data[f'n_q_branch={label}'] = (n_q_branch, None, [f'n_r_branch={label}'])

        else:
            if ('q0_' in label) or ('q1_' in label):
                data[f'n_r_branch={label}'] = (n_r_branch, None, None)
                data[f'n_q_branch={label}'] = (n_q_branch, None, [f'n_r_branch={label}'])   
            if cdim:
                data[f'n_c_branch={label}'] = (n_c_branch, None, [f'n_r_branch={label}'])
    
    if get_full:
        dat_package = (params, data, params_list, data_list, 
                        branches, dressed_evals, dressed_evecs)
        return dat_package
    else: 
        return params, data