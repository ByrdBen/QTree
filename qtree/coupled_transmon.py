import numpy as np
import scqubits as scq
import qutip as qt 
import re

class CoupledTransmon(object):
    def __init__(self, params):
        self.params = params

        # Set of allowed attributes
        attr = ['EJ', 'EC', 'ncut', 
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
        
        tmon = scq.Transmon(EJ=self.EJ, EC=self.EC, ng=0, ncut=self.ncut, truncated_dim=2*self.ncut+1, id_str="Tmon")
        resonator = scq.Oscillator(E_osc=self.f_r, truncated_dim=self.osc_trunc, id_str="Res")

        self.transmon = tmon
        self.resonator = resonator
        
        hs = scq.HilbertSpace([tmon, resonator])
        
        hs.add_interaction(
            expr=f"{self.g_n} * n * (a + adag)",  # g is directly inserted
            op1=("n", tmon.n_operator(), tmon),
            op2=("a", resonator.annihilation_operator(), resonator),
            op3=("adag", resonator.creation_operator(), resonator),
            add_hc=False
        )
        self.hs = hs
        
        H_full = hs.hamiltonian()
        self.H = H_full        
        
        # Precompute eigenvalues and lookup table
        #self.evals = self.H.eigenvals()
        self.hs.generate_lookup()  # generate the lookup table
        
    def eigenvals(self, count=8):
        return self.H.eigenenergies()[:count]
        
    def w01(self):
        E = self.eigenvals(2)
        return (E[1] - E[0])
    
    def anharmonicity(self):
        E = self.eigenvals(3)
        return (E[2] - E[1]) - (E[1] - E[0])
       
    def w01_n(self, n_photons=0):
        """01 transition frequency for given photon number in resonator."""
        idx_0n = self.hs.dressed_index((0, n_photons))
        idx_1n = self.hs.dressed_index((1, n_photons))
        max_idx = np.max([idx_0n, idx_1n])
        evals = self.eigenvals(max_idx+1)
        return evals[idx_1n] - evals[idx_0n]

    def chi01(self, n_photons=0):
        """
        Dispersive shift of the qubit transition
        for a given photon number in the resonator.
        """
        return self.w01_n(n_photons) - self.w01_n(0)