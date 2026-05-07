import sys
import gc
from tqdm import tqdm
from dataclasses import dataclass
from qtree.branch_analysis import *
from qtree.coupled_fluxonium import *
import scipy.constants as con
import os
import pickle
    
save_folder = f"/home/babyrd/branches/Personal/results/test/{sys.argv[6]}/" 
print("Save folder exists:", os.path.exists(save_folder), flush=True)
print("Python argv:", sys.argv)
print("Current dir:", os.getcwd())
print("Save folder:", save_folder)

chain_product = float(sys.argv[4])
chain_ratio   = float(sys.argv[5])

chain_trunc = 6

# Setting some vals
f_trunc    = int(sys.argv[1])
ncut       = f_trunc
osc_trunc  = int(sys.argv[2])
flux       = float(sys.argv[3])
lookup     = False

EJ            = 4.684
EL            = .491
EC            = 1.848
g_n           = .12
g_phi         = 1j * g_n
f_r           = 7.2
coupling_type = 'capacitive'
chain_mode    = True

#EJ_a    = np.sqrt(chain_product / chain_ratio)
#EC_a    = np.sqrt(chain_product * chain_ratio)
EJ_a    = 59.902        # GHz
cg_a    = .0256 * 1e-15 # farad
c_a     = 26.8 * 1e-15  # farad
num_JJ  = 122
g_chain = get_g_chain(EJ, EJ_a, .06, 
                      num_JJ=num_JJ, 
                      cg_a=cg_a, c_a=c_a)

EC_a = con.e**2 / (2*c_a) * 1.509*1e24
f_p  = np.sqrt(8 * EJ_a * EC_a)

k    = 1 * np.pi / num_JJ # assuming just a single array mode
con0 = 2*c_a*(1-np.cos(k)) 
f_c  = f_p * np.sqrt(con0 / (cg_a + con0)) # lowest array mode freq

fit_params = {}
# This may be excessive but it feels more flexible
# if things need to change
var_list = ['f_trunc', 'ncut', 'osc_trunc', 'flux', 'lookup', 'EJ', 'EL', 'EC', 
            'g_n', 'f_r', 'coupling_type', 'chain_mode', 'chain_trunc', 'g_chain',
            'f_c']

units = {'flux': r"$\Phi_0$"}
_locals = locals()

fit_params.update({
    name: (_locals[name], units.get(name), None)
    for name in var_list
})
    
fit_params['flux'] = (flux, r'$\Phi_0$', None)
dat_package, H_full = get_objs(fit_params, 'fluxonium')
params, data = branch_analysis(dat_package, update_flux=False)

# Path where you want to save all your files
os.makedirs(save_folder, exist_ok=True)  # create folder if it doesn't exist

# Example: save multiple files
filename = os.path.join(save_folder, f"res_flux_{flux:.4f}.pkl")
with open(filename, "wb") as f:
    pickle.dump(data, f)