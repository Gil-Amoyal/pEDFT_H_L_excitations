import psi4
from psi4.driver.procrouting.response.scf_response import tdscf_excitations
import time
import sys
import os
sys.path.append(os.path.abspath("/home/gilamo/git/pEDFT_HLgap/pEDFT"))
from LibPerturb import *

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

eV = 27.211324570273
basis = 'cc-pvdz'

#alpha = 0.25
#OMEGA = 0.24
#lc_wpbe0 = {
#    "name": "tuned_lc_wpbe0",
#    "x_functionals": {"GGA_X_HJS_PBE": {"omega": OMEGA , "alpha": 1-alpha}},
#    "x_hf": {"alpha": alpha , "beta": 1-alpha , "omega" : OMEGA},
#    "c_functionals": {"GGA_C_PBE": {}}
#}

def flatten(l):
    return [item for sublist in l for item in sublist]


# i is the molecule
def dftDATA(i):
 psi4.set_output_file(f'{i}'+"{FUN}_Psi4.out")
 psi4.set_memory(int(5e10))
 psi4.set_options({'basis': basis,
                   'scf_type': 'df',
	           'maxiter' : 500,
                   'reference' : 'rhf',
                   'save_jk' : True,
                   'df_ints_io' : 'save',
                   'e_convergence': 1e-8,
                   'd_convergence': 1e-8})

 xyz = '0 1 \n' + "".join(open(f'{i}'+'.xyz', 'r').readlines()[2:])
 xyz = psi4.geometry(xyz)
 t1 = time.time()
 sys.stdout = open(f"{i}_{FUN}_E.out", "w")
 print(f"{i}_{FUN}/{basis}")
 print("Starting SCF Calculation")

 e, wfn = psi4.energy('SCF', dft_functional='{FUN}', molecule=xyz, return_wfn=True)
 kh = wfn.nalpha()-1
 orbs = sorted(flatten([i.tolist() for i in wfn.epsilon_a().nph]))
 t2 = time.time()
 print("------KS------")
 print(f"SCF Energy: {e:7.10f} Hartree")
 print(f"KS First Excitation Energy: {(orbs[kh+1]-orbs[kh])*eV:7.10f} eV")
 print(f"KS First Excitation Wavelength: {1.240*10**-6/((orbs[kh+1]-orbs[kh])*eV)*10**9:7.2f} nm")

 LUMOrange = np.arange(0,1,1)
 pEDFT = {}
 pEDFTlist = []
 pEDFTEpsilon = {}
 Epsilonlist = []
 OscList = {}
 EGapPerturb = []
 XHelp = ExcitationHelper(wfn, RKS=False, FailonSCF=False)
 XHelp.SetMix(Mix=0.5, CMix=0.5)

 for l in LUMOrange:
     XHelp.SetTo(Shift=l)
     E = XHelp.SolveSingle()*eV
     pEDFT[l] = (E, XHelp.f01)
     pEDFTlist.append((l, E, XHelp.f01))
     pEDFTEpsilon[l] = XHelp.epsilonE*1.
     Epsilonlist.append((l, XHelp.epsilonE))
     EGapPerturb.append((l, XHelp.EGapPerturb))
     # Calculate other oscillators
     OscList[l] = [None]*len(LUMOrange)
     for lp in LUMOrange:
         OscList[l][lp] = XHelp.GetOscillator(k0=XHelp.kh, k1=XHelp.kl+lp)

 pEDFTlist = sorted(pEDFTlist, key = lambda t: t[2])
 t5 = time.time()

 print("-----------------")
 print("------pEDFT------") 
 print("-----------------")
 
 print("pEDFT First Excitation Energy:     " + str(pEDFT[0][0]) + "eV" )
 print("pEDFT First Excitation Wavelength: " + str((pEDFT[0][0])/1.240*10**-3) + " nm")
 print("HOMO-LUMO gap: " + str(pEDFT[0][0]) + "eV" )
 print("This is the oscillator strength of HOMO-LUMO gap:   " + str(pEDFT[0][1]))
 
 print("-----------HOMO-LUMO+n transitions-----------")
 print(pEDFT)
 print("-----Sorted based on Oscillator Strength-----")
 print(pEDFTlist)
 print(XHelp.f01)
 print(np.linalg.norm(XHelp.f01))
 print("-------------Oscillator Strength-------------")
 print(OscList)
 print("----------------------------------------------")
 print("-----------------pEDFTEpsilon-----------------")
 print("----------------------------------------------")

 import pandas as pd
 data = pEDFTEpsilon
 df = pd.DataFrame(data)
 pd.set_option('display.max_rows', None)
 print(df)

 print("----------------------------------------------")
 print("------------------EGapPerurb------------------")
 print("----------------------------------------------")
 print(EGapPerturb)
 print("----------------------------------------------")

 print("------Running Time------")
 print(f"SCF   time taken: {t2-t1:7.2f} s")
 print(f"pEDFT time taken: {t5-t4:7.2f} s")
 print(f"Total time taken: {t5-t1:7.2f} s")

#will calculate default if not specified in a different script python $.py -i molec -z func
from  optparse import OptionParser

genParse = OptionParser()
genParse.add_option('-i', type="string", default="ketene")
#genParse.add_option('-z', type="string", default="pbe50")

(options, args) = genParse.parse_args()

dftDATA(options.i)
