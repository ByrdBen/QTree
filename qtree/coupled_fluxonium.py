import numpy as np
import scipy.constants as con
import scqubits as scq
import qutip as qt 
import re

class CoupledFluxonium(object):
    """
    Generates coupled fluxonium/resonator hamiltonian.
    """
    def __init__(self, params):
        self.params = params

        # Set of allowed attributes
        attr = ['EJ', 'EC', 'EL', 'flux', 'ncut', 
                'f_trunc', 'f_r', 'g_n', 'g_phi', 'g_chain',
                'osc_trunc', 'chain_mode', 'chain_trunc', 'f_c', 'g_chain',
                'coupling_type']
        
        # Boolean to speed up computation regarding lookup table
        lookup = params['lookup'][0]
        for key, val in params.items():
            # Sick function that strips non-alphanumeric content from keys
            
            if 'E' in key:
                clean_key = re.sub(r'[^0-9a-zA-Z]', '', key)
            else:
                clean_key = re.sub(r'[^0-9a-zA-Z_]', '', key)
            # Don't wish to bog down class with too many params
            if clean_key in attr:
                # Set attributes of self to nice clean keys
                setattr(self, clean_key, val[0])
        
        # Make fluxonium
        fluxonium = scq.Fluxonium(EJ=self.EJ, EC=self.EC, EL=self.EL, flux=self.flux, 
                                  cutoff=self.ncut, truncated_dim=self.f_trunc, id_str="Fl")
        self.fluxonium = fluxonium
        
        # Make resonator
        resonator = scq.Oscillator(self.f_r, 1/np.sqrt(2), truncated_dim=self.osc_trunc, 
                                   id_str="Res")
        
        self.resonator = resonator

        if (self.chain_mode):
            # Make chain mode
            chain_mode = scq.Oscillator(self.f_c, 1/np.sqrt(2), truncated_dim=self.chain_trunc, 
                               id_str="Chain")
            
            self.chain_mode = chain_mode

            # Combine circuit elements
            hs = scq.HilbertSpace([fluxonium, resonator, chain_mode])

        else:
            # Combine circuit elements
            hs = scq.HilbertSpace([fluxonium, resonator])

        self.hs = hs

        # Turn on coupling
        if self.coupling_type in ("capacitive", "mixed"): 
            hs.add_interaction(
                expr=f"{self.g_n} * n * (a + adag)",  # g is directly inserted
                op1=("n", fluxonium.n_operator(), fluxonium),
                op2=("a", resonator.annihilation_operator(), resonator),
                op3=("adag", resonator.creation_operator(), resonator),
                add_hc=False
            )
        if self.coupling_type in ("inductive", "mixed"):
            _, _, coupling_ratio = self.phi_components()
            hs.add_interaction(
                expr=f"{self.g_n * coupling_ratio} * phi * (a - adag)",  # g is directly inserted
                op1=("phi", fluxonium.phi_operator(), fluxonium),
                op2=("a", resonator.annihilation_operator(), resonator),
                op3=("adag", resonator.creation_operator(), resonator),
                add_hc=False
            )

        if self.chain_mode:
            # Term 1
            self.hs.add_interaction(
                expr=f"{self.g_chain[0]} * cos_phi * (a + adag) ** 2",  # g is directly inserted
                op1=("cos_phi", self.fluxonium.cos_phi_operator(beta = 2*np.pi*self.flux), self.fluxonium),
                op2=("a", self.chain_mode.annihilation_operator(), self.chain_mode),
                op3=("adag", self.chain_mode.creation_operator(), self.chain_mode),
                add_hc=False
            )
            
            # Term 2
            self.hs.add_interaction(
                expr=f"{self.g_chain[1]} * sin_phi * (a + adag)",  # g is directly inserted
                op1=("sin_phi", self.fluxonium.sin_phi_operator(beta = 2*np.pi*self.flux), self.fluxonium),
                op2=("a", self.chain_mode.annihilation_operator(), self.chain_mode),
                op3=("adag", self.chain_mode.creation_operator(), self.chain_mode),
                add_hc=False
            )

        if self.coupling_type == "inductive_long":
            phi_long, _, coupling_ratio = self.phi_components()
            self.hs.add_interaction(
                expr=f"{self.g_n * coupling_ratio} * phi_long * (a - adag)",
                op1=("phi_long", phi_long, fluxonium),
                op2=("a", resonator.annihilation_operator(), resonator),
                op3=("adag", resonator.creation_operator(), resonator),
                add_hc=False
            )
            
        if self.coupling_type == "inductive_trans":
            _, phi_trans, coupling_ratio = self.phi_components()
            self.hs.add_interaction(
                expr=f"{self.g_n * coupling_ratio} * phi_trans * (a - adag)",
                op1=("phi_trans", phi_trans, fluxonium),
                op2=("a", resonator.annihilation_operator(), resonator),
                op3=("adag", resonator.creation_operator(), resonator),
                add_hc=False
            )
                    
        H_full = hs.hamiltonian()
        self.H = H_full        
        
        # Important note! Some functions will fail if lookup table not generated!  
        if lookup:
            # Precompute eigenvalues and lookup table
            self.hs.generate_lookup()
        
    def eigenvals(self, count=8):
        return self.H.eigenenergies()[:count]
        
    def w01(self):
        E = self.eigenvals(2)
        return (E[1] - E[0])
    
    def anharmonicity(self):
        E = self.eigenvals(3)
        return (E[2] - E[1]) - (E[1] - E[0])
       
    def w01_n(self, n_photons=0):
        """
        01 transition frequency for given photon number in resonator
        """
        if not self.lookup:
            print("Lookup Table Not Generated! Please Check Params!")
        else:
            idx_0n = self.hs.dressed_index((0, n_photons))
            idx_1n = self.hs.dressed_index((1, n_photons))
            max_idx = np.max([idx_0n, idx_1n])
            evals = self.hs.eigensys(max_idx+1)[0]
            return evals[idx_1n] - evals[idx_0n]
    
    def w02_n(self, n_photons=0):
        """
        02 transition frequency for given photon number in resonator
        """
        if not self.lookup:
            print("Lookup Table Not Generated! Please Check Params!")
        else:
            idx_0n = self.hs.dressed_index((0, n_photons))
            idx_2n = self.hs.dressed_index((2, n_photons))
            max_idx = np.max([idx_0n, idx_2n])
            evals = self.hs.eigensys(max_idx+1)[0]
            return evals[idx_2n] - evals[idx_0n]

    def chi01(self, n_photons=0):
        """
        0-1 dispersive shift at specific photon number
        """
        return self.w01_n(n_photons) - self.w01_n(0)
    
    def chi02(self, n_photons=0):
        """
        0-2 dispersive shift at specific photon number
        """
        return self.w02_n(n_photons) - self.w02_n(0)
    
    def get_qubit_drive(self):
        """
        Return fluxonium drive operator
        """
        a = qt.destroy(self.f_trunc)
        adag = a.dag()
        drive = qt.Qobj(adag - a)
        return qt.tensor(drive, qt.qeye(self.osc_trunc))
    
    def get_resonator_drive(self):
        """
        Return fluxonium drive operator
        """
        a = self.resonator.annihilation_operator()
        adag = self.resonator.creation_operator()
        drive_op = adag + a
        return qt.tensor(qt.qeye(self.f_trunc), qt.Qobj(drive_op))
    
    def get_n_operator(self):
        """
        Return fluxonium charge operator
        """
        n = self.fluxonium.n_operator()
        return qt.tensor(qt.Qobj(n), qt.qeye(self.osc_trunc))
    
    def update_flux(self, flux, lookup=False, print_update=False):
        """
        Update value of flux in fluxonium, need to rebuild hamiltonian
        Lookup table generation optional, computationally expensive
        """
        self.flux = flux
        self.fluxonium.flux = flux
        if self.chain_mode:
            self.hs = scq.HilbertSpace([self.fluxonium, self.resonator, self.chain_mode])
        else:
            self.hs = scq.HilbertSpace([self.fluxonium, self.resonator])
        
        
        # Turn on coupling
        if self.coupling_type in ("capacitive", "mixed"): 
            self.hs.add_interaction(
                expr=f"{self.g_n} * n * (a + adag)",  # g is directly inserted
                op1=("n", self.fluxonium.n_operator(), self.fluxonium),
                op2=("a", self.resonator.annihilation_operator(), self.resonator),
                op3=("adag", self.resonator.creation_operator(), self.resonator),
                add_hc=False
            )
        if self.coupling_type in ("inductive", "mixed"):
            _, _, coupling_ratio = self.phi_components()
            self.hs.add_interaction(
                expr=f"{self.g_n * coupling_ratio} * phi * (a - adag)",  # g is directly inserted
                op1=("phi", self.fluxonium.phi_operator(), self.fluxonium),
                op2=("a", self.resonator.annihilation_operator(), self.resonator),
                op3=("adag", self.resonator.creation_operator(), self.resonator),
                add_hc=False
            )
        if self.chain_mode:
            # Term 1
            self.hs.add_interaction(
                expr=f"{self.g_chain[0]} * cos_phi * (a + adag) ** 2",  # g is directly inserted
                op1=("cos_phi", self.fluxonium.cos_phi_operator(beta = 2*np.pi*self.flux), self.fluxonium),
                op2=("a", self.chain_mode.annihilation_operator(), self.chain_mode),
                op3=("adag", self.chain_mode.creation_operator(), self.chain_mode),
                add_hc=False
            )

            # Term 2
            self.hs.add_interaction(
                expr=f"{self.g_chain[1]} * sin_phi * (a + adag)",  # g is directly inserted
                op1=("sin_phi", self.fluxonium.sin_phi_operator(beta = 2*np.pi*self.flux), self.fluxonium),
                op2=("a", self.chain_mode.annihilation_operator(), self.chain_mode),
                op3=("adag", self.chain_mode.creation_operator(), self.chain_mode),
                add_hc=False
            )

        if self.coupling_type == "inductive_long":
            phi_long, _, coupling_ratio = self.phi_components()
            self.hs.add_interaction(
                expr=f"{self.g_n * coupling_ratio} * phi_long * (a - adag)",
                op1=("phi_long", phi_long, self.fluxonium),
                op2=("a", self.resonator.annihilation_operator(), self.resonator),
                op3=("adag", self.resonator.creation_operator(), self.resonator),
                add_hc=False
            )
            
        if self.coupling_type == "inductive_trans":
            _, phi_trans, coupling_ratio = self.phi_components()
            self.hs.add_interaction(
                expr=f"{self.g_n * coupling_ratio} * phi_trans * (a - adag)",
                op1=("phi_trans", phi_trans, self.fluxonium),
                op2=("a", self.resonator.annihilation_operator(), self.resonator),
                op3=("adag", self.resonator.creation_operator(), self.resonator),
                add_hc=False
            )
                 
        self.H = self.hs.hamiltonian()
        
        if lookup:
            self.hs.generate_lookup()
            
        if print_update:
            print(r'Flux set to ' + f'{flux:.2f} ' + r'$\Phi_0$')

    def phi_components(self):
        evals, evecs = self.fluxonium.eigensys(evals_count=self.f_trunc)
        phi_fock     = self.fluxonium.phi_operator()
        n_fock       = self.fluxonium.n_operator()
        
        # Transform phi to truncated eigenbasis
        phi_eig = evecs.T.conj() @ phi_fock @ evecs
        # Grab re-normalization constant
        phi_mat_elem = phi_eig[0 , 1]

        # Transform phi to truncated eigenbasis
        n_eig = evecs.T.conj() @ n_fock @ evecs
        # Grab re-normalization constant
        n_mat_elem = n_eig[0 , 1]

        # Renormalizing our inductive coupling strength
        coupling_ratio = 1j * (np.abs(n_mat_elem) / np.abs(phi_mat_elem))
        
        # Decompose
        phi_long_eig  = np.diag(np.diag(phi_eig))            # diagonal only
        phi_trans_eig = phi_eig - phi_long_eig                # off-diagonal only
        
        # Project back to Fock basis for add_interaction
        phi_long  = evecs @ phi_long_eig  @ evecs.T.conj()
        phi_trans = evecs @ phi_trans_eig @ evecs.T.conj()
        
        return phi_long, phi_trans, coupling_ratio


def get_g_chain(EJ, EJ_a, N, num_JJ=None, zpf=None, cg_a=None, c_a=None):
    # See notes
    if zpf is not None:
        g_chain2 = EJ * zpf * (N ** (1/2))
        g_chain1 = EJ / 2 * (zpf ** 2) * N
    else:
        EC_a     = con.e**2 / (2*c_a) * 1.509*1e24
        term1    = 2 * EC_a / EJ_a 
        term2    = 1 / (1 + (cg_a / c_a)*(num_JJ**2/(2**2 * np.pi**2)))
        zpf      = (term1 * term2) ** (1/4)
        g_chain2 = EJ * zpf * (N ** (1/2))
        g_chain1 = EJ / 2 * (zpf ** 2) * N
    
    return (g_chain1, g_chain2)