import psi4
import numpy as np
import scipy.linalg as la

import numpy.random as np_ra

eV = 27.211

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

##### Two-body integrals ####

class JKHelper:
    def __init__(self, wfn, omega=None, mem=None,
                 Debug=False):
        self.NBas = wfn.nmo()
        self.Has_RS = not(omega is None)
        
        if wfn.jk() is None:
            self.NewJK(wfn.basisset(), omega, mem=mem)
        else:
            #self.JK = wfn.jk()
            #self.JK.set_do_wK(True)
            #self.JK.initialize()
            wfn.jk().finalize()
            self.NewJK(wfn.basisset(), omega, mem=mem)

        self.Debug = Debug
            
    def NewJK(self, basis, omega, mem=None):
        self.JK = psi4.core.JK.build(basis, jk_type="DF",
                                     do_wK=self.Has_RS)
        if mem is None:
            mem = self.JK.memory_estimate()
            MaxMem = int(psi4.get_memory()*0.8)
            if mem>MaxMem:
                print("Need approximately 1024^%4.1f bytes out of 1024^%4.1f"\
                      %(np.log(mem)/np.log(1024), np.log(MaxMem)/np.log(1024) ))
                mem = MaxMem
                
        self.JK.set_memory( mem )
        self.JK.set_wcombine(False) # Comment out for older psi4
        if self.Has_RS:
            self.JK.set_omega(omega)
            self.JK.set_omega_alpha(0.0) # Comment out for older psi4
            self.JK.set_omega_beta(1.0) # Comment out for older psi4
            self.JK.set_do_wK(True)

        self.JK.initialize()
        

    def FJ(self, C, CR=None):
        return self.FMaster(C, CR, "J")
    def FK(self, C, CR=None):
        return self.FMaster(C, CR, "K")
    def FK_w(self, C, CR=None):
        return self.FMaster(C, CR, "K_w")
    
    def FMaster(self, C, CR=None, Mode='J'):
        if self.Debug: print("Getting Fock operator %s"%(Mode))
        if not(CR is None):
            if len(CR.shape)==1:
                CRM = psi4.core.Matrix.from_array(CR.reshape((self.NBas,1)))
            else:
                CRM = psi4.core.Matrix.from_array(CR)
            
        if len(C.shape)==1:
            CM = psi4.core.Matrix.from_array(C.reshape((self.NBas,1)))
        else:
            CM = psi4.core.Matrix.from_array(C)
            
        self.JK.C_clear()
        self.JK.C_left_add(CM)
        if CR is None:
            self.JK.C_right_add(CM)
        else:
            self.JK.C_right_add(CRM)
            
        self.JK.compute()
            
        if Mode.upper()=='J':
            return self.JK.J()[0].to_array(dense=True)
        elif Mode.upper()=='K':
            return self.JK.K()[0].to_array(dense=True)
        elif Mode.upper() in ('WK', 'KW', 'K_W'):
            if self.Has_RS:
                return self.JK.wK()[0].to_array(dense=True)
            else: return 0.
        else:
            return self.JK.J()[0].to_array(dense=True), \
                self.JK.K()[0].to_array(dense=True)
        


##### Process density-fitting ####
# THIS ROUTINE IS RETAINED BUT NEVER USED

################################################################################
# Note - all the ERI needs rewriting
# See ~/Molecules/Misc-Code/JK-Tests.py for help
################################################################################

def GetDensityFit(wfn, basis, mints, omega=None):
    aux_basis = psi4.core.BasisSet.build\
        (wfn.molecule(), "DF_BASIS_SCF", "",
         "RIFIT", basis.name())
    zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
    SAB = np.squeeze(mints.ao_eri(aux_basis, zero_basis, basis, basis))
    metric = mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis)
    metric.power(-0.5, 1e-14)
    metric = np.squeeze(metric)
    ERIA = np.tensordot(metric, SAB, axes=[(1,),(0,)])

    if not(omega is None):
        # Get the density fit business
        # Need to work out how to do density fit on rs part
        IntFac_Apq = psi4.core.IntegralFactory\
            (aux_basis, zero_basis, basis, basis)
        IntFac_AB  = psi4.core.IntegralFactory\
            (aux_basis, zero_basis, aux_basis, zero_basis)
        SAB_w = np.squeeze(
            mints.ao_erf_eri(omega, IntFac_Apq) )
        metric_w = mints.ao_erf_eri(omega, IntFac_AB )
        metric_w.power(-0.5, 1e-14)
        metric_w = np.squeeze(metric_w)
        
        # ERI in auxilliary - for speed up
        ERIA_w = np.tensordot(metric_w, SAB_w, axes=[(1,),(0,)])
    else:
        ERIA_w = None

    return ERIA, ERIA_w

##### This is a hack to convert a UKS superfunctional to its RKS equivalent
# Internal routine
# https://github.com/psi4/psi4/blob/master/psi4/driver/procrouting/dft/dft_builder.py#L251
sf_from_dict =  psi4.driver.dft.build_superfunctional_from_dictionary
# My very hacky mask
def sf_RKS_to_UKS(DFA):
    DFA_Dict = { 'name':DFA.name()+'_u'}
    DFA_Dict['x_functionals']={}
    DFA_Dict['c_functionals']={}
    for x in DFA.x_functionals():
        Name = x.name()[3:]
        alpha = x.alpha()
        omega = x.omega()

        if np.abs(alpha)>1e-5:
            if omega==0.:
                DFA_Dict['x_functionals'][Name] = {"alpha": alpha, }
            else:
                DFA_Dict['x_functionals'][Name] = {"alpha": alpha, "omega": omega, }
    for c in DFA.c_functionals():
        Name = c.name()[3:]
        alpha = c.alpha()
        omega = c.omega()

        if np.abs(alpha)>1e-5:
            if omega==0.:
                DFA_Dict['c_functionals'][Name] = {"alpha": alpha, }
            else:
                DFA_Dict['c_functionals'][Name] = {"alpha": alpha, "omega": omega, }


    npoints = psi4.core.get_option("SCF", "DFT_BLOCK_MAX_POINTS")
    DFAU, _ = sf_from_dict(DFA_Dict,npoints,1,False)
    return DFAU
##### End hack

# For nice debug printing
def NiceArr(X):
    return "[ %s ]"%(",".join(["%8.3f"%(x) for x in X]))
def NiceArrInt(X):
    return "[ %s ]"%(",".join(["%5d"%(x) for x in X]))
def NiceMat(X):
    N = X.shape[0]
    if N==0:
        return "[]"
    elif N==1:
        return "["+NiceArr(X[0,:])+"]"
    elif N==2:
        return "["+NiceArr(X[0,:])+",\n "+NiceArr(X[1,:])+"]"
    else:
        R = "["
        for K in range(N-1):
            R+=NiceArr(X[K,:])+",\n "
        R+=NiceArr(X[N-1,:])+"]"
        return R

# One-spot rs-hybrid handler
def RSHybridParams(alpha, beta, omega):
    # alpha E_x + beta E_x^{lr} + beta E_x^{sr-DFA} + (1-alpha-beta) E_x^{DFA}
    WDFA = 1. - alpha - beta
    WDFA_SR = beta
    WHF = alpha
    WHF_LR = beta

    # See C:\Users\tgoul\Dropbox\Collabs\Ensemble\EGKS\Implementation\psi4-Notes.pdf
    
    return WDFA, WDFA_SR, WHF, WHF_LR

# Handle PBE0_XX calculations
def GetDFA(DFA):
    if DFA[:5].lower()=="pbe0_":
        X = DFA.split('_')
        alpha = float(X[1])/100.
        if len(X)>2: f_c = max(float(X[2])/100.,1e-5)
        else: f_c = 1.
        return {
            'name':DFA,
            'x_functionals': {"GGA_X_PBE": {"alpha": 1.-alpha, }},
            'c_functionals': {"GGA_C_PBE": {"alpha": f_c, }},
            'x_hf': {"alpha": alpha, },
        }
    elif DFA[:5].lower()=="pbe_h":
        return {
            'name':DFA,
            'x_functionals': {"GGA_X_HJS_PBE": {"alpha": 1., "omega": 10., }},
            'c_functionals': {"GGA_C_PBE": {"alpha": 1., }},
            'x_hf': {"alpha": 0., },
        }
    elif DFA[:5].lower()=="wpbe_":
        # Format
        # wpbe_[alpha%]_[omega]_[beta%]_[corr%]_[lda%]
        #
        # Only alpha needs to be specificiec - others default to:
        #    omega=0.3, beta=1-alpha, corr=100%, lda=0%
        #
        # E.g. wpbe_25_0.5 gives alpha=0.25, omega=0.5, beta=0.75, corr=1.0, lda=0.0
        
        
        X = DFA.split('_')
        alpha = float(X[1])/100.
        if len(X)>2: omega = float(X[2])
        else: omega = 0.3
        if len(X)>3: beta = float(X[3])/100
        else: beta = 1.-alpha
        if len(X)>4: WC = float(X[4])/100
        else: WC = 1.
        if len(X)>5: WLDA_SR = float(X[4])/100
        else: WLDA_SR = 0.

        WDFA, WDFA_SR, WHF, WHF_LR = RSHybridParams(alpha, beta, omega)
        
        DFADef =  {
            'name':DFA,
            'x_hf': {"alpha":WHF, "beta":WHF_LR, "omega":omega, }, 
            'x_functionals': {"GGA_X_HJS_PBE": {"alpha":WDFA_SR - WLDA_SR, "omega":omega, }, },
            'c_functionals': {"GGA_C_PBE": {"alpha":WC, } },
        }
        if np.abs(WDFA)>1e-5:
            DFADef["x_functionals"]["GGA_X_PBE"] = {"alpha":WDFA, }
        if np.abs(WLDA_SR)>1e-5:
            DFADef["x_functionals"]["LDA_X_ERF"] = {"alpha":WLDA_SR, "omega":omega, }
            print("Short-range LDA does not appear to be implemented in psi4")
            quit()
        return DFADef
    else:
        return DFA

# Get the degeneracy of each orbital
def GetDegen(epsilon, eta=1e-5):
    Degen = np.zeros((len(epsilon),),dtype=int)
    for k in range(len(epsilon)):
        ii =  np.argwhere(np.abs(epsilon-epsilon[k])<eta).reshape((-1,))
        Degen[k] = len(ii)
    return Degen

#################################################################################################
# This code handles degeneracies detected and used by psi
#################################################################################################

class SymHelper:
    def __init__(self, wfn):
        self.NSym = wfn.nirrep()
        self.NBasis = wfn.nmo()
        
        self.eps_so = wfn.epsilon_a().to_array()
        self.C_so = wfn.Ca().to_array()
        self.ao_to_so = wfn.aotoso().to_array()
        
        if self.NSym>1:
            self.eps_all = np.hstack(self.eps_so)
            self.k_all = np.hstack([ np.arange(len(self.eps_so[s]), dtype=int)
                                       for s in range(self.NSym)])
            self.s_all = np.hstack([ s * np.ones((len(self.eps_so[s]),), dtype=int)
                                       for s in range(self.NSym)])
        else:
            self.eps_all = self.eps_so * 1.
            self.k_all = np.array(range(len(self.eps_all)))
            self.s_all = np.zeros((len(self.eps_all),), dtype=int)

        self.ii_sorted = np.argsort(self.eps_all)
        self.eps_sorted = self.eps_all[self.ii_sorted]
        self.k_sorted = self.k_all[self.ii_sorted]
        self.s_sorted = self.s_all[self.ii_sorted]

        self.ks_map = {}
        for q in range(len(self.ii_sorted)):
            self.ks_map[(self.s_sorted[q], self.k_sorted[q])] = q

    # Do a symmetry report to help identifying orbitals
    def SymReport(self, kh, eta=1e-5):
        epsh = self.eps_sorted[kh] + eta
        print("Orbital indices by symmetry - | indicates virtual:")
        for s in range(self.NSym):
            Str = "Sym%02d : "%(s)
            eps = self.eps_so[s]
            if not(hasattr(eps, '__len__')) or len(eps)==0: continue

            kk_occ = []
            kk_unocc = []
            for k, e in enumerate(eps):
                if e<epsh: kk_occ += [ self.ks_map[(s,k)] ]
                else: kk_unocc += [ self.ks_map[(s,k)] ]

            Arr = ["%3d"%(k) for k in kk_occ] + [" | "] \
                + ["%3d"%(k) for k in kk_unocc]
            if len(Arr)<=16:
                print("%-8s"%(Str) + " ".join(Arr))
            else:
                for k0 in range(0, len(Arr), 16):
                    kf = min(k0+16, len(Arr))
                    if k0==0:
                        print("%-8s"%(Str) + " ".join(Arr[k0:kf]))
                    else:
                        print(" "*8 + " ".join(Arr[k0:kf]))

    # Report all epsilon
    def epsilon(self):
        return self.eps_sorted

    # Report a given orbital, C_k
    def Ck(self, k):
        if self.NSym==1:
            return self.C_so[:,k]
        else:
            s = self.s_sorted[k]
            j = self.k_sorted[k]

            return self.ao_to_so[s].dot(self.C_so[s][:,j])

    # Report all C
    def C(self):
        if self.NSym==1:
            return self.C_so * 1.
        else:
            C = np.zeros((self.NBasis, self.NBasis))
            k0 = 0
            for k in range(self.NSym):
                C_k = self.ao_to_so[k].dot(self.C_so[k])
                dk = C_k.shape[1]
                C[:,k0:(k0+dk)] = C_k
                k0 += dk
            return C[:,self.ii_sorted]

    # Convert the so matrix to dense form
    def Dense(self, X):
        if self.NSym==1:
            return X
        else:
            XX = 0.
            for s in range(self.NSym):
                XX += self.ao_to_so[s].dot(X[s]).dot(self.ao_to_so[s].T)
            return XX

    # Solve a Fock-like equation using symmetries
    # if k0>0 use only the subspace spanned by C[:,k0:]
    def SolveFock(self, F, S=None, k0=-1):
        # Note, k0>=0 means to solve only in the basis from C[:,k0:]
        if self.NSym==1:
            if k0<=0:
                return la.eigh(F, b=S)
            else:
                # FV = SVw
                # V=CU
                # FCU = SCUw
                # (C^TFC)U = (C^TSC)Uw
                # XU = Uw
                
                C = self.C()[:,k0:]
                F_C = (C.T).dot(F).dot(C)
                w, U = la.eigh(F_C)
                return w, C.dot(U)
        else:
            k0s = [0]*self.NSym
            ws = [None]*self.NSym
            Cs = [None]*self.NSym

            if k0>0:
                # Use no terms
                for s in range(self.NSym):
                    k0s[s] = self.NBasis
                    
                # Evaluate the smallest k value for each symmetry
                for i in range(k0,self.NBasis):
                    s = self.s_sorted[i]
                    k0s[s] = min(k0s[s],self.k_sorted[i])

            for s in range(self.NSym):
                # Project onto the subset starting at k0s
                C_ao = self.ao_to_so[s].dot(self.C_so[s][:,k0s[s]:])
                F_C = (C_ao.T).dot(F).dot(C_ao)
                if not(S is None):
                    S_C = (C_ao.T).dot(S).dot(C_ao)
                else: S_C = None

                if F_C.shape[0]>0:
                    ws[s], Us = la.eigh(F_C, b=S_C)
                    Cs[s] = C_ao.dot(Us)
                else:
                    ws[s] = []
                    Cs[s] = [[]]

            # Project back onto the main set
            k0 = max(k0,0)
            w = np.zeros((self.NBasis - k0,)) + 200. # If errors, make sure they're high energy
            C = np.zeros((self.NBasis,self.NBasis - k0))
            
            #for i in self.ii_sorted[k0:]: # Old and I am sure not correct
            for i in range(k0, self.NBasis): # Index of orbital
                s = self.s_sorted[i] # Its symmetry
                k = self.k_sorted[i] # Its k value in the symmetry
                w[i-k0] = ws[s][k-k0s[s]] # Copy in - k0s is the smallest value in the subset
                C[:,i-k0] = Cs[s][:,k-k0s[s]] # Like above
                
            return w, C
        

#################################################################################################
# This is the main pEDFT code
#################################################################################################

class ExcitationHelper:
    def __init__(self, wfn, RKS=True, wfn0=None,
                 FailonSCF = True,
                 Report=10):
        self.Report = Report
        
        self.wfn = wfn
        self.RKS = RKS
        self.FailonSCF = FailonSCF
        
        if wfn0 is None:
            wfn0 = wfn

        self.SymHelp = SymHelper(wfn0)
        
        self.Da = self.SymHelp.Dense(wfn0.Da().to_array())
        self.epsilon = self.SymHelp.epsilon()
        self.C = self.SymHelp.C()
        self.F = self.SymHelp.Dense(wfn0.Fa().to_array())

        self.epsilonE = self.epsilon * 1.
        self.CE = self.C * 1.

        basis = wfn.basisset()
        self.basis = basis
        self.nbf = self.wfn.nmo() # Number of basis functions
        self.NAtom = self.basis.molecule().natom()

        if not(basis.has_puream()):
            print("Must use a spherical basis set, not cartesian")
            print("Recommend rerunning with def2 or cc-type basis set")
            quit()
       

        self.Enn = self.basis.molecule().nuclear_repulsion_energy()
        
        self.mints = psi4.core.MintsHelper(self.basis)
        self.S_ao = self.mints.ao_overlap().to_array(dense=True)
        self.T_ao = self.mints.ao_kinetic().to_array(dense=True)
        self.V_ao = self.mints.ao_potential().to_array(dense=True)
        self.H_ao = self.T_ao + self.V_ao


        # These are used for DFA calculations
        self.VPot = wfn.V_potential() # Note, this is a VBase class
        try:
            self.DFA = self.VPot.functional()
        except:
            self.DFA = None
            self.VPot = None

        # Ensure we have UKS for the excitations
        if not(self.DFA is None) and \
           not(self.RKS) and (wfn0==wfn):
            # Convert DFA from RKS to UKS
            self.DFAU = sf_RKS_to_UKS(self.DFA)
            # Make a new VPot for the UKS DFA
            self.VPot = psi4.core.VBase.build(self.VPot.basis(), self.DFAU, "UV")
            self.VPot.initialize()

        # Work out the range-separation and density functional stuff
        self.xDFA, self.xDFA_w = 0., 0.
        self.omega = None
        self.alpha = 0.
        self.beta = 0.
        self.xi, self.xi_Full, self.xi_RS = 0., 0., 0.
        if not(self.DFA is None):
            # Implement DFAs
            if self.DFA.is_x_hybrid():
                # Hybrid functional
                self.alpha = self.DFA.x_alpha()
                self.xDFA = 1. - self.alpha
                if self.DFA.is_x_lrc():
                    # Range-separated hybrid
                    self.omega = self.DFA.x_omega()
                    self.beta = self.DFA.x_beta()

                    WDFA, WDFA_SR, WHF, WHF_LR = RSHybridParams(self.alpha, self.beta, self.omega)
                    if Report>=0:
                        print("# RS hybrid alpha = %.2f, beta = %.2f, omega = %.2f"\
                              %(self.alpha, self.beta, self.omega),
                              flush=True)

                    self.xDFA = 1. - self.alpha # 1. - self.alpha
                    self.xDFA_w = - self.beta # - self.beta
            else:
                # Conventional functional
                self.xDFA = 1.
                
            #print("*"*72)
            #print(self.xDFA, self.xDFA_w, self.alpha, self.beta, self.omega)
            #print("*"*72)
            
            # psi4 matrices for DFA evaluation
            self.DMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.DMb = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMa = psi4.core.Matrix(self.nbf, self.nbf)
            self.VMb = psi4.core.Matrix(self.nbf, self.nbf)
        else:
            self.xDFA = 0. # Pure HF theory

        self.Has_RS = not(self.omega is None)
        #self.ERIA, self.ERIA_w = GetDensityFit(self.wfn, self.basis, self.mints, self.omega)

        self.JKHelp = JKHelper(wfn, self.omega)

        # Dipole operator: 3 set of matrices
        self.Di_ao = [X.to_array(dense=True) for X in self.mints.ao_dipole()]
        # Gradient of the overlap: NAtom x 3 set of matrices
        self.dS_ao = [[X.to_array(dense=True) for X in self.mints.ao_oei_deriv1("OVERLAP", K)] \
                      for K in range(self.NAtom)]
            
        self.NBas = wfn0.nmo()
        self.NOcc = wfn0.nalpha()
        self.kh = wfn0.nalpha()-1
        self.kl = self.kh+1
        self.Degen = GetDegen(self.epsilon)

        self.kFrom = self.kh
        self.kTo = self.kl

        self.Exc_gs, self.Exc_ts, self.Exc_dx = None, None, None
        
        if self.SymHelp.NSym>1 and self.Report>=0:
            self.SymHelp.SymReport(self.kh)


        if self.Report>0:
            print("eps = %s eV"%(NiceArr(self.epsilon[max(0,self.kh-2):min(self.NBas,self.kl+3)]*eV)))
            print("eps_h = %8.3f/%8.2f, eps_l = %8.3f/%8.2f [Ha/eV]"\
                  %(self.epsilon[self.kh], self.epsilon[self.kh]*eV,
                    self.epsilon[self.kl], self.epsilon[self.kl]*eV,),
                  flush=True)

        self.FHF = self.H_ao*1.

        J = self.JKHelp.FJ(self.C[:,:(self.kh+1)])
        K = self.JKHelp.FK(self.C[:,:(self.kh+1)])
        self.FHF += 2.*J - K
        
        #for I in range(self.kh+1):
        #    CI = self.C[:,I] 
        #    self.FHF += 2.*self.GetFJ(CI) - self.GetFK(CI)

        # Mixing parameters
        self.Mix = None
        self.Mix2 = None
        self.CMix = None

        if self.Report>1:
            if self.omega is None:
                print("# xDFA = %.2f - standard hybrid"%(self.xDFA))
            else:
                print("# xDFA = %.2f, xDFA_sr = %.2f, omega = %.2f"\
                      %(self.xDFA, self.xDFA_w, self.omega))

    # Choose the orbital to promote from using degeneracy and shift (=0 for highest)
    def SetFrom(self, Degen=0, Shift=0, SymType=None):
        if not(SymType is None):
            s = self.SymHelp.s_sorted
            k = np.arange(len(s))
            kk = k[s==SymType]
            if len(kk)<1 or len(kk[kk<=self.kh])<1:
                print("Symmetry %d does not exist for from"%(SymType))
                quit()

            self.kFrom = kk[kk<=self.kh].max()
            return self.kFrom
        
        if Degen<1:
            self.kFrom=self.kh-Shift
            return self.kFrom
        
        Count = 0
        self.kFrom = -1
        for k in range(self.kh,-1,-1):
            if self.Degen[k]==Degen:
                if Count==Shift:
                    self.kFrom=k
                    break
                Count+=1
                
        if self.kFrom<0:
            print("# There is no orbital compatible with Degen=%d and Shift=%d"%(Degen, Shift))
            quit()

        if Degen>1:
            self.kFromAll = np.argwhere(np.abs(
                self.epsilon-self.epsilon[self.kFrom])<1e-6).reshape((-1,))
            
        return self.kFrom
    
    # Choose the orbital to promote to using degeneracy and shift (=0 for lowest)
    def SetTo(self, Degen=0, Shift=0, SymType=None):
        if not(SymType is None):
            s = self.SymHelp.s_sorted
            k = np.arange(len(s))
            kk = k[s==SymType]

            if len(kk)<1 or len(kk[kk>=self.kl])<1:
                print("Symmetry %d does not exist for to"%(SymType))
                print(s, k)
                quit()
                
            self.kTo = kk[kk>=self.kl].min()
            return self.kTo
        
        if Degen<1:
            self.kTo=self.kl+Shift
            return self.kTo
        
        Count = 0
        self.kTo = -1
        for k in range(self.kl,self.NBas):
            if self.Degen[k]==Degen:
                if Count==Shift:
                    self.kTo=k
                    break
                Count+=1
        if self.kTo<0:
            print("# There is no orbital compatible with Degen=%d and Shift=%d"%(Degen, Shift))
            quit()

        return self.kTo

    def SetMix(self, Mix, Mix2=None, CMix=None):
        if Mix2 is None:
            Mix2 = Mix

        self.Mix = Mix
        self.Mix2 = Mix2
        self.CMix = CMix

    # Provide a guessed set of excited state orbitals
    def SetCGuess(self, C):
        CVir = C[:,self.kl:]*1.
        COcc = self.CE[:,:self.kl]

        O = (CVir.T).dot(self.S_ao).dot(COcc)
        for k in range(O.shape[0]):
            CVir[:,k] -= COcc.dot(O[k,:])
            CVir[:,k] /= np.sqrt(1. - min(0.9,np.sum(O[k,:]**2)))

        #print((CVir.T).dot(self.S_ao).dot(CVir))

        self.CE[:,self.kl:] = CVir
            

    def GetFJ(self, CI, Pre=1.):
        if np.abs(Pre)<1e-5: return 0.
        else: return Pre*self.JKHelp.FJ(CI)
        #A = np.tensordot(self.ERIA,CI,axes=((2,),(0,)))
        #A = np.tensordot(A,CI,axes=((1,),(0,)))
        #return np.tensordot(A,self.ERIA,axes=((0,),(0,)))

    def GetFK(self, CI, Pre=1.):
        if np.abs(Pre)<1e-5: return 0.
        else: return Pre*self.JKHelp.FK(CI)
        #A = np.tensordot(self.ERIA,CI,axes=((2,),(0,)))
        #return np.tensordot(A,A,axes=((0,),(0,)))

    def GetFJ_w(self, CI, Pre=1.):
        if np.abs(Pre)<1e-5: return 0.
        print("****  Calling RS FJ -- weird! ****")
        quit()
        #if self.ERIA_w is None: return 0.
        #A = np.tensordot(self.ERIA_w,CI,axes=((2,),(0,)))
        #A = np.tensordot(A,CI,axes=((1,),(0,)))
        #return Pre*np.tensordot(A,self.ERIA_w,axes=((0,),(0,)))

    def GetFK_w(self, CI, Pre=1.):
        if np.abs(Pre)<1e-5: return 0.
        else: return Pre*self.JKHelp.FK_w(CI)
        #if self.ERIA_w is None: return 0.
        #A = np.tensordot(self.ERIA_w,CI,axes=((2,),(0,)))
        #return np.tensordot(A,A,axes=((0,),(0,)))

    # Compute Hartree and exchange integrals
    def GetEJ(self, D):
        A = np.tensordot(self.ERIA,D,axes=((1,2),(1,0)))
        return 0.5*np.dot(A,A)

    def GetEK(self, D):
        B = np.tensordot(self.ERIA,D,axes=((1,),(0,)))
        B = np.tensordot(self.ERIA,B,axes=((0,1),(0,2)))
        return 0.5*np.tensordot(D,B)
    
    # Compute range-separated Hartree and exchange integrals
    def GetEJ_w(self, D):
        if self.ERIA_w is None: return 0.
        A = np.tensordot(self.ERIA_w,D,axes=((1,2),(1,0)))
        return 0.5*np.dot(A,A)

    def GetEK_w(self, D):
        if self.ERIA_w is None: return 0.
        B = np.tensordot(self.ERIA_w,D,axes=((1,),(0,)))
        B = np.tensordot(self.ERIA_w,B,axes=((0,1),(0,2)))
        return 0.5*np.tensordot(D,B)

    # Compute energies using occupation factors
    def GetEMaster_Occ(self, f, C=None, Mode='J'):
        if C is None: C = self.CE

        C = C[:,:len(f)]
        CR = C * f[None,:]

        F = self.JKHelp.FMaster(C, CR, Mode)
        if isinstance(F, np.ndarray):
            return 0.5*np.einsum('pk,pq,qk', C, F, CR)
        else:
            return 0.

    def EJ_Occ(self, f, C=None):
        return self.GetEMaster_Occ(f, C, 'J')

    def EK_Occ(self, f, C=None):
        return self.GetEMaster_Occ(f, C, 'K')

    def EK_w_Occ(self, f, C=None):
        return self.GetEMaster_Occ(f, C, 'wK')



    # Compute the DFA terms
    def GetFDFA(self, C0, C1, Return="DV", Da=None, Double=False):
        # Set Return value to:
        #   DV gives Vts - Vgs (default for unknown option)
        #   E gives Egs, Ets
        #   V gives Vgs, Vts
        #   EV or VE gives Egs, Ets, Vgs, Vts

        # Use internal Da by default
        if Da is None: Dgs = self.Da
        else: Dgs = Da
        
        # Singlet gs
        if not(self.RKS):
            self.DMa.np[:,:] = Dgs
            self.DMb.np[:,:] = Dgs
            self.VPot.set_D([self.DMa,self.DMb])
            self.VPot.compute_V([self.VMa,self.VMb])
            Egs = self.VPot.quadrature_values()["FUNCTIONAL"]
            Vgs = self.VMa.to_array(dense=True)
            
            self.DMa.np[:,:] = Dgs + np.outer(C1,C1)
            self.DMb.np[:,:] = Dgs - np.outer(C0,C0)
            self.VPot.set_D([self.DMa,self.DMb])
            self.VPot.compute_V([self.VMa,self.VMb])
            Ets = self.VPot.quadrature_values()["FUNCTIONAL"]
            Vts = self.VMa.to_array(dense=True)
        else:
            self.DMa.np[:,:] = Dgs - np.outer(C0,C0)
            self.VPot.set_D([self.DMa,])
            self.VPot.compute_V([self.VMa,])
            Egs = self.VPot.quadrature_values()["FUNCTIONAL"]
            Vgs = self.VMa.to_array(dense=True)
            
            self.DMa.np[:,:] = Dgs + np.outer(C1,C1)
            self.VPot.set_D([self.DMa,])
            self.VPot.compute_V([self.VMa,])
            Ets = self.VPot.quadrature_values()["FUNCTIONAL"]
            Vts = self.VMa.to_array(dense=True)

        if Double:
            self.DMa.np[:,:] = Dgs + np.outer(C1,C1) - np.outer(C0,C0)
            self.DMb.np[:,:] = Dgs + np.outer(C1,C1) - np.outer(C0,C0)
            self.VPot.set_D([self.DMa,self.DMb])
            self.VPot.compute_V([self.VMa,self.VMb])
            self.Exc_dx = self.VPot.quadrature_values()["FUNCTIONAL"]


        self.Exc_gs, self.Exc_ts = Egs, Ets
            
        if Return.upper()=="E":
            return Egs, Ets
        elif Return.upper()=="V":
            return Vgs, Vts
        elif Return.upper() in ("EV", "VE"):
            return Egs, Ets, Vgs, Vts
        else:
            return Vts-Vgs


    def GuessLowestTriple(self, Range=3):
        MinGap = 10000
        for kFrom in range(self.kh, max(0,self.kh-Range)-1, -1):
            for kTo in range(self.kl, min(self.NBas, self.kl+Range)):
                Gap = self.SolveTriple(k0=kFrom, k1=kTo, Silent=True, MaxStep=0)
                if (Gap<MinGap):
                    MinGap = Gap
                    self.kFrom = kFrom
                    self.kTo = kTo
        if self.Report>0:
            print("# Lowest gap from %d to %d"%(self.kFrom, self.kTo))
        return self.kFrom, self.kTo

    def GuessLowestSingle(self, Range=3):
        MinGap = 10000
        for kFrom in range(self.kh, max(0,self.kh-Range)-1, -1):
            for kTo in range(self.kl, min(self.kl+Range, self.NBas)):
                Gap = self.SolveSingle(k0=kFrom, k1=kTo, Silent=True, MaxStep=0)
                if (Gap<MinGap):
                    MinGap = Gap
                    self.kFrom = kFrom
                    self.kTo = kTo
        if self.Report>0:
            print("# Lowest gap from %d to %d"%(self.kFrom, self.kTo))
        return self.kFrom, self.kTo

    def SolveTriple(self, *args, **kwargs):
        return self.SolveGeneral(J1 = [-1, 0], K1 = [ 0, 0],
                                 J1_w = [0, 0], K1_w = [ 0, 0],
                                 **kwargs)
    def SolveSingle(self, *args, xi=0., RS=True, **kwargs):
        self.xi = xi
        if RS:
            self.xi_Full = min(-xi, 0)
            self.xi_RS = xi
        else:
            self.xi_Full = -np.abs(xi)
            self.xi_RS = 0.
            
        return self.SolveGeneral(J1 = [-1, 0], K1 = [2*(1+self.xi_Full), 0],
                                 J1_w = [0, 0], K1_w = [2*self.xi_RS, 0],
                                 **kwargs)

    def SolveDouble(self, *args, xi=0., RS=True, **kwargs):
        self.xi = xi
        if RS:
            self.xi_Full = min(-xi, 0)
            self.xi_RS = xi
        else:
            self.xi_Full = -np.abs(xi)
            self.xi_RS = 0.
            
        return self.SolveGeneral(J1 = [-2, 1], K1 = [1*(1+self.xi_Full), 0],
                                 J1_w = [0, 0], K1_w = [1*self.xi_RS, 0],
                                 **kwargs)

    def SolveGeneral(self, k0=None, k1=None,
                     J1 = [-1, 0], K1 = [ 2, 0],
                     J1_w = [0, 0], K1_w = [ 0, 0],
                     SHx = 1., SDFA=1.,
                     kFreeze = None,
                     UseHF = False,
                     GapFactor = 1.,
                     ErrCut = 1e-5, MaxStep=80,
                     ShowOrbital = False,
                     Report = False,
                     Silent = False,
                     Debug = False, # This should be set to false normally
    ):
        if self.Report<=-1: Silent=True # very low report overrides silent

        if kFreeze is None:
            kFreeze = self.kl
        
        # Handle the new From and To
        if not(k0 is None) or not(k1 is None):
            if not(Silent):
                print("# The direct use of k0 and k1 is not recommended")
                print("# - use SetFrom and SetTo before calling for excitations")
            if not(k0 is None): k0=k0
            else: k0=self.kFrom
            if not(k1 is None): k1=k1
            else: k1=self.kTo
            Nk0 = 1
        else:
            k0 = self.kFrom
            k1 = self.kTo
            if not(Silent):
                print("# From %3d to %3d"%(k0, k1))

            # Setting Nk0>1 will trigger a more complicated method that should be
            # strictly symmetry - but it doesn't appear to be necessary
            Nk0 = 1 # self.Degen[k0]
            if Nk0>1:
                k0All = self.kFromAll
            
        dk1 = k1-self.kl

        if Debug:
            C_In = self.CE*1.

        if Nk0 == 1:
            C0All = [self.CE[:,k0]*1.]
        else:
            C0All = []
            for k in k0All:
                C0All += [self.CE[:,k]*1.]
            
        C1 = self.CE[:,k1]*1.
        CX = self.CE[:,kFreeze:]*1.

        epsilon_old = self.epsilonE*1.
        epsilon_new = 1.*epsilon_old

        #####################################################################
        # Correct for DFA exchange
        # Note, the exchange acts only on the Fock term for l-l interactions
        # and is negative
        # But we _subtract_ the Fock in any DFA/hybrid so we have a positive
        # correction
        if not(UseHF or self.DFA is None):
            K1[1] += self.xDFA # This is a -xDFA times the -1 term
            K1_w[1] += self.xDFA_w
        #####################################################################

        # Show the perturbative correction
        if not(Silent):
            print("# HF correction = %6.3f v_{J,h} + %6.3f v_{K,h} + %6.3f v_{J,l} + %6.3f v_{K,l}"\
                  %(J1[0], K1[0], J1[1], K1[1]), flush=True)
            if self.Has_RS:
                print("# rs-HF correc. = %6.3f v_{J,h} + %6.3f v_{K,h} + %6.3f v_{J,l} + %6.3f v_{K,l}"\
                      %(J1_w[0], K1_w[0], J1_w[1], K1_w[1]), flush=True)
                

        DF1 = 0.
        for C0 in C0All:
            # Convert to operator
            DF1 += SHx * (self.GetFJ(C0, Pre=J1[0]) + self.GetFK(C0, Pre=K1[0]) \
                          + self.GetFJ(C1, Pre=(J1[1]+K1[1]))) \
                +  SHx * (self.GetFJ_w(C0, Pre=J1_w[0]) + self.GetFK_w(C0, Pre=K1_w[0]) \
                          + self.GetFK_w(C1, Pre=(J1_w[1]+K1_w[1])))

            
            # Add DFA if appropriate
            if not(UseHF or self.DFA is None):
                FDFA = self.GetFDFA(C0,C1)
                DF1 += SDFA*FDFA
                # TEMP
                t00 = (C0).dot(FDFA).dot(C0)
                t11 = (C1).dot(FDFA).dot(C1)
                #print("HYBRID TEST DFA %8.3f %8.3f"%(t00, t11))
                # END TEMP
                
        DF1 /= len(C0All)


        GapELUMO = ((C1).dot(DF1).dot(C1))
        #print("HYBRID TEST deps Gap %8.3f"%(GapELUMO))

        # Project onto C1
        DEP = C1.dot(DF1).dot(C1)
        self.EGapPerturb = GapFactor* (self.epsilon[k1]-self.epsilon[k0]+DEP)
        if not(Silent):
            print("# DFA initial correction = %8.3f Ha = %8.2f eV"\
                  %(DEP, DEP*eV))
            print("# Perturbative gap = %8.2f eV"%(self.EGapPerturb*eV), flush=True)
        if MaxStep==0:
            self.epsilonE = self.epsilon*1.
            self.epsilonE[kFreeze:]+=DEP
            return self.EGapPerturb

        # print(XHelp.EGapPerturb)
        
        # Introduce some smoothing
        FOld = self.F * 1.
        FOld2 = self.F * 1.
        C1Old = C1*1.

        Err = 1e3 # Make sure large error if no iterations
        if self.Mix is None:
            Mix = 0.3+0.3*self.xDFA
            Mix2 = 0.3*self.xDFA
        else:
            Mix = self.Mix
            Mix2 = self.Mix2

        print("Mix = %.3f, Mix2 = %.3f "%(Mix, Mix2), end="")
        if self.CMix is None: print("CMix = None")
        else: print("CMix = %.3f"%(self.CMix))
           
        eps1Old = 0.

        # Anderson initialisation
        DoAnderson = np.abs(Mix)>10.
        if DoAnderson:
            N_And = k1+1 - self.kl
            w_And = [ None, None, None ] # For Anderson mixing on _two_
            F_And = [ None, None, None ] # For Anderson mixing on _two_

        def Heaviside(x):
            if x<0.: return -1.
            else: return 1.
            
        for step in range(MaxStep): # If it's not done in MaxStep it's probably oscillating
            F1 = 0.
            for C0 in C0All:
                DF1 = SHx*(self.GetFJ(C0, Pre=J1[0]) + self.GetFK(C0, Pre=K1[0]) \
                           + self.GetFJ(C1, Pre=(J1[1]+K1[1]))) \
                    + SHx*(self.GetFJ_w(C0, Pre=J1_w[0]) + self.GetFK_w(C0, Pre=K1_w[0]) \
                           + self.GetFK_w(C1, Pre=(J1_w[1]+K1_w[1]))) # Can use the same term for both 
                if UseHF:
                    F1 += self.FHF + DF1
                elif self.DFA is None:
                    F1 += self.F + DF1
                else:
                    #####################################################################
                    # Here, form:
                    # D_avg = 2\sum_{i<=h}outer(C_i,C_i)
                    # D_up = \sum_{i<=l}outer(C_i,C_i)
                    # D_down = \sum_{i<h}outer(C_i,C_i)
                    # and use the code from LibEnsemble ==> GetFDFA to calculate
                    # DFDFA = V(D_up,D_down) - V(D_avg,D_avg)

                    DFDFA = SDFA*self.GetFDFA(C0,C1, Return="DV")
                    F1 += self.F + DF1 + DFDFA
            F1 /= len(C0All)

            if self.CMix is None:
                FNew = (1.-Mix2)*( (1.-Mix)*F1 + Mix*FOld ) + Mix2*FOld2
                w, CX = self.SymHelp.SolveFock(FNew, S=self.S_ao, k0=kFreeze)

                FOld2 = FOld*1.
                FOld = FNew*1.
            else:
                FNew = F1*1.
                w, CX = self.SymHelp.SolveFock(FNew, S=self.S_ao, k0=kFreeze)
                C1 = CX[:,dk1]

                w1NoMix = w[dk1]*1.
                C1p = Heaviside(C1.dot(self.S_ao).dot(C1Old)) * C1Old
                
                C1New = (1. - self.CMix)*C1 + self.CMix*C1p
                C1New /= np.sqrt( (C1New).dot(self.S_ao).dot(C1New) )
                CX[:,dk1] = C1New
                w[dk1] = (C1New).dot(F1).dot(C1New)

                w1Old = (C1Old).dot(F1).dot(C1Old)

                C1Old = C1New * 1.

                if self.Report>3:
                    print("%3d %10.5f %10.5f %10.5f"%(step, w[dk1], w1NoMix, w1Old), flush=True)

            i1 = dk1
            
            C1 = CX[:,i1]

            epsilon_new[kFreeze:]=w

            Err = GapFactor*np.abs((epsilon_new[k1]-epsilon_new[k0])-(epsilon_old[k1]-epsilon_old[k0]))
            
            if Report or (step>(MaxStep*2/3)):
                # Show that the index shifted if it did
                if not(i1==dk1):
                    if not(Silent):
                        print("MOM shift @ %d : %d to %d"%(step, dk1, i1))
                if not(Silent):
                    print("%3d: epsilon(%d) = %.3f, epsilon(%d) = %.3f, Gap = %.3f [%.5f]"\
                          %(step,
                            k0, epsilon_new[k0],
                            k1, epsilon_new[k1],
                            GapFactor*(epsilon_new[k1]-epsilon_new[k0]),
                            Err), flush=True)
            epsilon_old = 1.*epsilon_new
            if ((Err<ErrCut) and (step>3)) or MaxStep<=1:
                break
        
        if Err<ErrCut:
            if not(Silent):
                print("# Took %d steps to get to sc Err = %.6f"%(step, Err), flush=True)
            # Now that we have successfully converged things, we need to
            # compute the other virtual orbitals using the correct operator.
            # This replaces J_1 with K_1, where appropriate.
            DF1 = SHx*(self.GetFJ(C0, Pre=J1[0]) + self.GetFK(C0, Pre=K1[0]) \
                       + self.GetFJ(C1, Pre=J1[1]) + self.GetFK(C1, Pre=K1[1])) \
                + SHx*(self.GetFJ_w(C0, Pre=J1_w[0]) + self.GetFK_w(C0, Pre=K1_w[0]) \
                       + self.GetFK_w(C1, Pre=(J1_w[1]+K1_w[1])))
            # Add DFA if appropriate
            if not(UseHF or self.DFA is None):
                DF1 += SDFA*self.GetFDFA(C0,C1)
            FFinal = self.F + DF1

            w, CX = self.SymHelp.SolveFock(FFinal, S=self.S_ao, k0=kFreeze)

            self.CE[:,kFreeze:] = CX
            self.epsilonE = epsilon_new
        else:
            print("# WARNING! Self-consistency cycle failed. Value returned is the perturbative gap\n"\
                  + "# not the self-consistent pEDFT gap. Orbitals left unchanged.")
            print("# Debug : Steps = %3d, Err = %.3f"%(step, Err), flush=True)
            if self.FailonSCF:
                quit()
                
            self.epsilonE = self.epsilon*1.
            self.epsilonE[kFreeze:]+=DEP

        if ShowOrbital:
            P = np.dot(C1, (self.S_ao).dot(self.C))**2
            Report = []
            for K, p in enumerate(P):
                if p>5e-3:
                    Report += ["%3.0f%% of %3d"%(100.*p, K)]
            print("pEDFT orbital |%3d> is: %s"%(self.kTo, "; ".join(Report)))

        if Debug:
            O = (C_In.T).dot(self.S_ao).dot(self.CE)
            print("Overlaps")
            print(NiceMat(O[k0:(k1+3),k0:(k1+3)]**2))

        # Get EST = 2(hl|lh) -- the singlet/triplet splitting
        self.EST = 2.*np.einsum('p,q,pq',C0,C0,self.GetFK(C1))

        # Get E01 = <S0|H|S1>
        # 1-RDM part
        self.E01 = (C0).dot(self.H_ao.dot(C1))
        # 2-RDM part
        for i in range(self.kh):
            A = 2.*self.GetFJ(self.CE[:,i]) - self.GetFK(self.CE[:,i])
            self.E01 += C0.dot(A.dot(C1))
        self.E01 = np.abs(self.E01) # real orbitals so may always take as +ve
        if not(Silent):
            print("# EST = %5.2f E01 = %7.4f"%(self.EST*eV, self.E01*eV), flush=True)

        # Energy of transition
        self.DE01 = GapFactor*(self.epsilonE[k1]-self.epsilonE[k0])

        # Oscillator strength
        self.f01 = self.GetOscillator()

        # Gradient of overlaps (not working)
        #self.dS01 = np.array([[C0.dot(X).dot(C1) for X in Y] for Y in self.dS_ao])
        
        if self.Report>0:
            print("eps = %s eV"%(NiceArr(self.epsilonE[max(0,self.kh-2):min(self.NBas,self.kl+3)]*eV)))
            print("eps_h = %8.3f/%8.2f, eps_l = %8.3f/%8.2f [Ha/eV]"\
                  %(self.epsilonE[self.kh], self.epsilonE[self.kh]*eV,
                    self.epsilonE[self.kl], self.epsilonE[self.kl]*eV,),
                  flush=True)
                  
        return self.DE01

    # Use the current orbitals to compute energies
    # Call for |0>=|gs> immediately after initalisation
    # Call for |1>=|ts> after SolveTriple
    # Call for |2>=|ss> and |3>=|dx> after SolveSingle
    def GetTotalEn(self, State=0, k0=None, k1=None, Force=False):
        if k0 is None: k0, C0 = self.kFrom, self.CE[:,self.kFrom]
        else: C0 = self.CE[:,k0]
            
        if k1 is None: k1, C1 = self.kTo, self.CE[:,self.kTo  ]
        else: C1 = self.CE[:,k1]

        fa = np.zeros((k1+1,))
        fb = np.zeros((k1+1,))
        
        fa[:self.NOcc] = 1.
        fb[:self.NOcc] = 1.

        self.DFrom = np.outer(C0, C0)
        self.DTo   = np.outer(C1, C1)

        self.GetFDFA(C0, C1, Double=True)
            
        if State == 1:
            Da = self.Da + np.outer(C1, C1)
            Db = self.Da - np.outer(C0, C0)
            DT = Da + Db
            
            Ts = np.vdot(DT, self.T_ao)
            EV = np.vdot(DT, self.V_ao)

            #EH = self.GetEJ(DT)
            #Ex = -self.GetEK(Da) -self.GetEK(Db)
            #Ex_w = -self.GetEK_w(Da) -self.GetEK_w(Db)
            
            fa[k1] = 1.
            fb[k0] = 0.
            fT = fa + fb
            
            EH = self.EJ_Occ(fT)
            Ex = - self.EK_Occ(fa) - self.EK_Occ(fb)
            Ex_w = - self.EK_w_Occ(fa) - self.EK_w_Occ(fb)

            EHxc = EH + self.alpha*Ex + self.beta*Ex_w + self.Exc_ts

            E = Ts + EV + EHxc

            if not(Force):
                return E+self.Enn, (Ts, EV, EH, EHxc-EH)

            self.DMa.np[:,:] = Da+Db
        elif State == 2:
            E1, (Ts, EV, EH, Exc) = self.GetTotalEn(State=1)

            EST = 2.*(C1).dot(self.GetFK(C0)).dot(C1)
            EST_w = 2.*(C1).dot(self.GetFK_w(C0)).dot(C1)

            ESTp = EST + self.xi_Full*EST + self.xi_RS*EST_w
            #if self.xi>0.: ESTp = EST - self.xi*(EST-EST_w)
            #else: ESTp = EST + self.xi*EST_w

            E = Ts + EV + EH + Exc + ESTp

            if not(Force):
                return E+self.Enn, (Ts, EV, EH+ESTp, Exc)

            self.DMa.np[:,:] = 2*self.Da +  + np.outer(C1, C1) - np.outer(C0, C0)
        elif State == 3:
            DT = 2.*(self.Da + np.outer(C1, C1) - np.outer(C0, C0))
            
            Ts = np.vdot(DT, self.T_ao)
            EV = np.vdot(DT, self.V_ao)

            fa[k1] = 1.
            fa[k0] = 0.
            fT = 2*fa
            
            #EH = self.GetEJ(DT)
            EH = self.EJ_Occ(fT)

            # Combination rule
            _, (a, b, c, Exc0) = self.GetTotalEn(State=0)
            _, (a, b, c, Exc1) = self.GetTotalEn(State=1)
            Exc_C = 2.*Exc1 - Exc0

            # Direct (on DFA only)
            Exc_D = Exc_C - 2.*self.Exc_ts + self.Exc_gs + self.Exc_dx

            # Mix
            #Exc = (1.-self.xi)*Exc_C + self.xi*Exc_D
            Exc = Exc_C

            # EHxc
            EHxc = EH + Exc
            
            EST = 2.*(C1).dot(self.GetFK(C0)).dot(C1)
            EST_w = 2.*(C1).dot(self.GetFK_w(C0)).dot(C1)

            ESTp = EST + self.xi_Full*EST + self.xi_RS*EST_w
            #if self.xi>0.: ESTp = EST - self.xi*(EST-EST_w)
            #else: ESTp = EST + self.xi*EST_w

            E = Ts + EV + EHxc + ESTp

            if not(Force):
                return E+self.Enn, (Ts, EV, EH+ESTp, EHxc-EH)

            self.DMa.np[:,:] = DT
        else: # State == 0
            Ts = 2.*np.vdot(self.Da, self.T_ao)
            EV = 2.*np.vdot(self.Da, self.V_ao)
            
            #EH = 4.*self.GetEJ(self.Da)
            #Ex = -2.*self.GetEK(self.Da)
            #Ex_w = -2.*self.GetEK_w(self.Da)
            EH = 4.*self.EJ_Occ(fa)
            Ex = -2.*self.EK_Occ(fa)
            Ex_w = -2.*self.EK_w_Occ(fa)

            EHxc = EH + self.alpha*Ex + self.beta*Ex_w + self.Exc_gs

            E = Ts + EV + EHxc

            if not(Force):
                return E+self.Enn, (Ts, EV, EH, EHxc-EH)

            self.DMa.np[:,:] = 2.*self.Da

        if Force:
            Fnn = self.basis.molecule().nuclear_repulsion_energy_deriv1([0,0,0]).to_array(dense=True)
            NAtom = Fnn.shape[0]

            # This doesn't work
            F = self.mints.potential_grad(self.DMa).to_array(dense=True)
            #print(F)
            #print(Fnn)
                
                
            return E + self.Enn, F + Fnn
        else:
            # Should never reach this point
            print("Something went wrong")
            quit()

    # Get the oscillator strength
    # Defaults to current gap but k0 and k1 can be specified
    # AsVector reports x, y and z components separately
    def GetOscillator(self, k0=None, k1=None,
                      DE01=None, # Defaults to epsilon gap
                      AsVector=False):
        if k0 is None: k0 = self.kFrom
        if k1 is None: k1 = self.kTo
        
        # Energy of transition
        if DE01 is None:
            DE01 = self.epsilonE[k1]-self.epsilonE[k0]

        C0 = self.CE[:,k0]
        C1 = self.CE[:,k1]
        
        # Oscillator strength (maybe working)
        Di01 = np.array([C0.dot(X).dot(C1) for X in self.Di_ao])
        if AsVector:
            return 2/3*DE01*Di01**2
        else:
            return 2/3*DE01*np.sum(Di01**2)

    # Get overlap of pEDFT and DFT orbitals
    def GetOverlaps(self, Show=10, Report=True):
        O = np.einsum('pj,pq,qk->jk', self.CE[:,self.kl:], self.S_ao, self.C[:,self.kl:])
        for i in range(Show):
            k = np.argmax(np.abs(O[i,:]))
            O[i,:] *= np.sign(O[i,k])
            if Report:
                W = O[i,:]**2
                q = np.argsort(-W)
                CW = 0.
                OStr = ""
                for k in q:
                    CW += W[k]
                    OStr += "%+6.3f|L+%d> "%(O[i,k], k)
                    if CW>0.99: break
                
                print("|EL+%2d> : %s"%(i, OStr))
            
        return O
    
    # Get spectrum and oscillators
    def GetSpectrum(self, Show=20, Report=True):
        Gap = []
        FT = []
        for k0 in range(self.kl):
            Gap += list(self.epsilonE[self.kl:]-self.epsilonE[k0])
            FT += [ (k0,k) for k in range(self.kl, len(self.epsilonE)) ]

        ii = np.argsort(Gap)
        ii = ii[:Show]
        Out = np.zeros((Show,4))
        for q,i in enumerate(ii):
            k0, k1 = FT[i]
            E = eV*Gap[i]
            O = self.GetOscillator(k0, k1)
            Out[q,:] = [k0, k1, Gap[i], O]

            if Report:
                print("From HOMO-%2d to LUMO+%2d at %7.2f with osc %.4f"\
                      %(self.kh-k0, k1-self.kl, E, O))
        return Out

    # Get a shitty approximation for experimental data
    def GetExperiment(self, lw = 10, Range=None):
        Out = self.GetSpectrum(Show=30, Report=False)
        E = Out[:,2]*eV
        O = Out[:,3]
        
        if Range is None:
            Range = [1240/E.max()-20, 1240/E.min()+20]
            
        X = np.linspace(Range[0], Range[1], 201)
        Y = 0.*X
        for k in range(len(E)):
            x0 = 1240/E[k]
            Y += O[k] * np.exp(-(X-x0)**2/10**2)

        return X, Y
            
if __name__ == "__main__":
    psi4.set_output_file("__perturb.dat")
    psi4.set_options({
        'basis' : 'def2-tzvp',
        'reference': 'rhf',
    })
    psi4.geometry("""
0 1
Be
symmetry c1""")

    E, wfn = psi4.energy("scf", return_wfn=True)
    
    XHelp = ExcitationHelper(wfn)
    Gap1 = XHelp.SolveSingle()
    Gap2 = XHelp.SolveDouble()

    
    print("Gap(sx) = %.2f, Gap(dx) = %.2f"%(Gap1*eV, Gap2*eV))
