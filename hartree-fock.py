#! python

import sys #system module
import numpy as np #numerical python, "np" is a standard-arbitrary abbreviation for numpy
import scipy as sp #numerical python, "np" is a standard-arbitrary abbreviation for numpy
from scipy import special

class PGF:
    """
    Primitive Gaussian Function
    """

    def __init__(self,exponent,center):
        """
        Constructs a PGF
        INPUT: (1) Exponent: float. (2) Center: np.array 
        OUTPUT: An object of type PGF 
        """
        self.exponent=exponent
        self.center=center
        self.normalization=1.0 #This value must have an initial number, otherwise overlap is missing a variable
        self.normalization=self.norm_para()

    def norm_para(self):
        """
        Calculates the normalization paramater of a CGF
        INPUT: A CGF
        OUTPUT: norm_para
        """
    
        overlap_sum=overlap_int(self,self)
        return 1.0/np.sqrt(overlap_sum)

def overlap_int(PGFa,PGFb):
    """
    Calculates the overlap integral between two s type PGF 
    INPUT: two PGFs
    OUTPUT: Overlap integral
    """

    alpha=PGFa.exponent
    beta=PGFb.exponent

    Ra=PGFa.center
    Rb=PGFb.center

    N_PGFa=PGFa.normalization
    N_PGFb=PGFb.normalization

    norm=np.linalg.norm(Ra-Rb)
    p=(alpha * beta) / (alpha + beta)
    K=(np.pi / (alpha + beta))**(1.5)

    return N_PGFa*N_PGFb*K*np.exp(-p*(norm**2))

def kinetic_int(PGFa,PGFb):
    """
    Calculates the kinetic integral between two s type PGF 
    INPUT: two PGFs
    OUTPUT: Kinetic integral
    """

    alpha=PGFa.exponent
    beta=PGFb.exponent

    Ra=PGFa.center
    Rb=PGFb.center

    N_PGFa=PGFa.normalization
    N_PGFb=PGFb.normalization

    p=(alpha * beta) / (alpha + beta)
    norm=np.linalg.norm(Ra-Rb)
    q=3 - (2*p*(norm**2))  
    K=(np.pi / (alpha + beta))**(1.5)
    return N_PGFa*N_PGFb*p*q*K*np.exp(-p*(norm**2)) 

def nuc_elec_int(PGFa,PGFb,center,Z):
    """
    Calculates the nuclear-electron integral between two s type PGF 
    INPUT: two PGFs,nuclear-center,charge
    OUTPUT: Nuclear Attraction integral
    """

    alpha=PGFa.exponent
    beta=PGFb.exponent

    Ra=PGFa.center
    Rb=PGFb.center

    N_PGFa=PGFa.normalization
    N_PGFb=PGFb.normalization
    
    m=((-2*np.pi)/(alpha + beta))
    p=(alpha * beta) / (alpha + beta)
    norm=np.linalg.norm(Ra-Rb)
    Rp=(alpha*Ra + beta*Rb)/(alpha + beta)
    norm2=np.linalg.norm(Rp-center)
    t=(alpha + beta)*(norm2)**2
    
    if (np.linalg.norm(Rp-center) > 1e-8):
        Ft=0.5*np.sqrt(np.pi/t)*sp.special.erf(np.sqrt(t))
    else:
        Ft=1.0

    return N_PGFa*N_PGFb*m*Z*np.exp(-p*(norm**2))*Ft

class CGF:
    """
    Contracted Gaussian Function 
    """

    def __init__(self,coefficients,PGFs):
        """
        Constructs a CGF
        INPUT: (1) Coeffcient: np.array. (2) PGFs: list of PGF 
        OUTPUT: An object of type CGF
        """

        if coefficients.size != len(PGFs):
            print('Error lengt of PGFS != lenght of coefficients')
            raise Error   

        self.coefficients=coefficients
        self.PGFs=PGFs
        self.normalization=1.0 #This values must have an initial number, otherwise overlap is missing a variable
        self.normalization=self.norm_para()

    def norm_para(self):
        """
        Calculates the normalization paramater of a CGF
        INPUT: A CGF
        OUTPUT: norm_para
        """
    
        overlap_sum=overlap(self,self)
        return 1.0/np.sqrt(overlap_sum)

def overlap(CGFa,CGFb):
    """
    Calculates the overlap integral between two s type CGFs 
    INPUT: two CGFs
    OUTPUT: Overlap integral
    """

    L_CGFa=CGFa.coefficients.size
    L_CGFb=CGFb.coefficients.size

    N_CGFa=CGFa.normalization
    N_CGFb=CGFb.normalization

    overlap_sum = 0.0

    for i in range(0, L_CGFa):
        for j in range(0, L_CGFb):
            overlap_sum += CGFa.coefficients[i] * CGFb.coefficients[j] * overlap_int(CGFa.PGFs[i],CGFb.PGFs[j])
    
    return N_CGFa*N_CGFb*overlap_sum

def kinetic(CGFa,CGFb):
    """
    Calculates the overlap integral between two s type CGFs 
    INPUT: two CGFs
    OUTPUT: Overlap integral
    """

    L_CGFa=CGFa.coefficients.size
    L_CGFb=CGFb.coefficients.size

    N_CGFa=CGFa.normalization
    N_CGFb=CGFb.normalization

    kinetic_sum = 0.0

    for i in range(0, L_CGFa):
        for j in range(0, L_CGFb):
            kinetic_sum += CGFa.coefficients[i] * CGFb.coefficients[j] * kinetic_int(CGFa.PGFs[i],CGFb.PGFs[j])
    
    return N_CGFa*N_CGFb*kinetic_sum

def nuc_elec(CGFa,CGFb,H_dist,Z):
    """
        Calculates the overlap integral between two s type CGFs
        INPUT: two CGFs, Nuclear-array
        OUTPUT: Overlap integral
        """
    
    L_CGFa=CGFa.coefficients.size
    L_CGFb=CGFb.coefficients.size
    L_H_dist=len(H_dist)

    N_CGFa=CGFa.normalization
    N_CGFb=CGFb.normalization
    
    nuc_elec_sum = 0.0
    
    for m in range (0, L_H_dist):
        for i in range(0, L_CGFa):
            for j in range(0, L_CGFb):
                nuc_elec_sum += CGFa.coefficients[i] * CGFb.coefficients[j] * nuc_elec_int(CGFa.PGFs[i],CGFb.PGFs[j],H_dist[m],Z)
    
    return N_CGFa*N_CGFb*nuc_elec_sum


if __name__ == '__main__':

    H_dist = []
    H_dist.append(np.array([0.0,0.0,0.7]))
    H_dist.append(np.array([0.0,0.0,-0.7]))

    P_basis = []
    P_basis.append(PGF(0.168856,H_dist[0]))
    P_basis.append(PGF(0.623913,H_dist[0]))
    P_basis.append(PGF(3.425250,H_dist[0]))
    P_basis.append(PGF(0.168856,H_dist[1]))
    P_basis.append(PGF(0.623913,H_dist[1]))
    P_basis.append(PGF(3.425250,H_dist[1]))

    CGF1=CGF(np.array([0.444635,0.535328,0.154329]),[P_basis[0],P_basis[1],P_basis[2]])
    CGF2=CGF(np.array([0.444635,0.535328,0.154329]),[P_basis[3],P_basis[4],P_basis[5]])

    S=np.zeros((2,2))
    S[0,0]=overlap(CGF1,CGF1)
    S[0,1]=overlap(CGF1,CGF2)
    S[1,0]=overlap(CGF2,CGF1)
    S[1,1]=overlap(CGF2,CGF2)

    K=np.zeros((2,2))
    K[0,0]=kinetic(CGF1,CGF1)
    K[0,1]=kinetic(CGF1,CGF2)
    K[1,0]=kinetic(CGF2,CGF1)
    K[1,1]=kinetic(CGF2,CGF2)

    VNe=np.zeros((2,2))
    VNe[0,0]=nuc_elec(CGF1,CGF1,H_dist,1)
    VNe[0,1]=nuc_elec(CGF1,CGF2,H_dist,1)
    VNe[1,0]=nuc_elec(CGF2,CGF1,H_dist,1)
    VNe[1,1]=nuc_elec(CGF2,CGF2,H_dist,1)
    
    Hcore=np.zeros((2,2))
    Hcore[0,0]=np.sum(VNe[0,0] + K[0,0])
    Hcore[0,1]=np.sum(VNe[0,1] + K[0,1])
    Hcore[1,0]=np.sum(VNe[1,0] + K[1,0])
    Hcore[1,1]=np.sum(VNe[1,1] + K[1,1])
    
    print ("S=",S)
    #print("K=",K)
    #print("VNe=",VNe)
    print("HCore=",Hcore)

#    K=np.zeros((6,6))
#
#    for i in range(0, 6):
#        for j in range(0, 6):
#            K[i,i] = kinetic_int(P_basis[i],P_basis[i])
#            K[j,j] = kinetic_int(P_basis[j],P_basis[j])
#            K[i,j] = kinetic_int(P_basis[i],P_basis[j])
#            
#    print(K)





