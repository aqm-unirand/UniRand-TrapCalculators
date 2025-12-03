#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:57:01 2023

@author: liyang
"""

import sys

import numpy as np

from arc import Lithium6, CG
from scipy.constants import physical_constants

from numpy.linalg import eigh

uB = physical_constants["Bohr magneton in Hz/T"][0]

# plt.style.use('/Users/liyang/Documents/Utilities/qlib/standardmini.mplstyle')
# plt.style.use('/Users/liyang/Documents/Utilities/qlib/helvet.mplstyle')

def diagcat(M1, M2):
    if M1.shape[0] != M1.shape[1]:
        return M2
    else:
        return np.block([[M1,np.zeros((M1.shape[0],M2.shape[1]))], [np.zeros((M2.shape[0], M1.shape[1])), M2]])

# def renorm(M, eta=1e-1):
#     M = np.multiply(np.abs(M)>1e-1,M)
#
#     for i in range(M.shape)
#
#     return M

uB = physical_constants["Bohr magneton in Hz/T"][0]
# uB = physical_constants["Bohr magneton in Hz/T"][0]
epsilon_0 = physical_constants["vacuum electric permittivity"][0]
hbar = physical_constants["reduced Planck constant"][0]


class Lithium6(Lithium6):
    I = 1
    wavelen = [671e-9, 321e-9]
    Afs = [6.701891e9, 1.92e9]
    Gamma = [2*np.pi*5.87e6, 2*np.pi*.754e6]
    Ahfs = [[(17375000.00, 0.), (-1155000.00, 0.)], [(5.3e6, 0), (-400000.00, 0)]]

    def _spinInit(self, j1, j2):
        js = np.arange(j1+j2,np.abs(j1-j2)-1,-1)
        jz = np.array([])
        j = np.array([])
        for ji in js:
            jz = np.append(jz,-np.arange(-ji,ji+1))
            j = np.append(j,ji*np.ones(int(2*ji+1)))

        J = np.matrix(np.diag(j))
        Jz = np.matrix(np.diag(jz))
        return J, Jz
    
    def _spinMatrices(self, j: float):
        """Generates spin-matrices for spin S

        The Sx,Sy,Sz spin matrices calculated using raising and lowering
        operators.

        Args:
            j (_type_): _description_

        Returns:
            array: [Sx,Sy,Sz]=SPINMATRICES(S)
        """
        mj = -np.arange(-j + 1, j + 1)
        jm = np.sqrt(j * (j + 1) - mj * (mj + 1))
        Jplus = np.matrix(np.diag(jm, 1))  # Raising Operator
        Jminus = np.matrix(np.diag(jm, -1))  # Lowering Operator
        Jx = (Jplus + Jminus) / 2.0
        Jy = (-Jplus + Jminus) * 1j / 2.0
        Jz = (Jplus * Jminus - Jminus * Jplus) / 2.0
        # J2=Jx**2+Jy**2+Jz**2
        return Jx, Jy, Jz

    def _spinCoupling(self, j1, j2):

        J, Jz = self._spinInit(j1, j2)

        # j1z = np.diag(-np.arange(-j1,j1+1))
        # j2z = np.diag(-np.arange(-j2,j2+1))
        # J1z = np.kron(j1z,np.eye(j2z.shape[0]))
        # J1 = j1*np.eye(J1z.shape[0])
        # J2z = np.kron(np.eye(j1z.shape[0]),j2z)
        # J2 = j2 * np.eye(J2z.shape[0])

        [j1x,j1y,j1z] = self._spinMatrices(j1)
        j1i = np.eye(j1z.shape[0])
        [j2x,j2y,j2z] = self._spinMatrices(j2)
        j2i = np.eye(j2z.shape[0])

        J1x = np.kron(j1x, j2i)
        J1y = np.kron(j1y, j2i)
        J1z = np.kron(j1z, j2i)
        J1 = np.kron(j1*j1i,j2i)
        J2x = np.kron(j1i, j2x)
        J2y = np.kron(j1i, j2y)
        J2z = np.kron(j1i,j2z)
        J2 = np.kron(j1i, j2i*j2)

        JJ = J1x*J2x + J1y*J2y + J1z*J2z


        return J, Jz, J1, J1z, J2, J2z, JJ

    def _basisTrans(self, j1, j2, kw='coupling'):
        J, Jz, J1, J1z, J2, J2z, JJ = self._spinCoupling(j1, j2)
        M = np.zeros(J.shape)

        for i in range(M.shape[1]):
            for j in range(M.shape[0]):
                c = CG(J1[j,j], J1z[j,j], J2[j,j], J2z[j,j], J[i,i], Jz[i,i])
                M[j,i] = c
        if kw == 'coupling':
            return (np.matrix(M), JJ)
        elif kw == 'z':
            return (np.matrix(M), J1z, J2z)


    def FS(self, n, l, s=0.5):

        F = np.matrix([[]])
        Fz = np.matrix([[]])
        J = np.matrix([[]])
        H = np.matrix([[]])
        if l == 0.:
            Afs = 0.
        else:
            Afs = self.Afs[n-2]
        for j in np.arange(l+s, np.abs(l-s)-1, -1):
            if l == 0:
                Ahfs = 152.1e6
            else:
                Ahfs = self.Ahfs[n-2][int(j-0.5)][0]
            Fj, Fjz = self._spinInit(self.I, j)
            Hj =  Ahfs/2 * (np.multiply(Fj, Fj+1) - np.eye(Fj.shape[0])*(self.I*(self.I+1)+j*(j+1))) + Afs/2*np.eye(Fj.shape[0])*(j*(j+1)-l*(l+1)-s*(s+1))
            F = diagcat(F,Fj)
            Fz = diagcat(Fz,Fjz)
            J = diagcat(J,np.eye(Fj.shape[0])*j)
            H = diagcat(H,Hj)

        return (H, F, Fz, J)



    # def MagneticInt(self, n, l, s=0.5, B=0):
    #     J, Jz, L, Lz, S, Sz, LS = self._spinCoupling(l,s)
    #     n = J.shape[0]
    #     M = np.zeros(J.shape)
    #     for i in range(M.shape[1]):
    #         for j in range(M.shape[0]):
    #             c = CG(L[j,j], Lz[j,j], S[j,j], Sz[j,j], J[i,i], Jz[i,i])
    #             M[j,i] = c
    #
    #
    #
    #
    #     return F[1:,1:], Fz[1:,1:], H[1:,1:]

    def M1int(self, l, B=0., s=0.5):

        M1, Lz, Sz = self._basisTrans(l,s,kw='z')
        M1, LS = self._basisTrans(l, s)

        Ii = np.eye(int(2*self.I+1))

        M1 = np.kron(M1, Ii)
        Lz = np.kron(Lz, Ii)
        Sz = np.kron(Sz, Ii)
        LS = np.kron(LS, Ii)

        M2 = np.array([[]])
        for j in np.arange(l+s,np.abs(l-s)-1,-1):
            Mj, _ = self._basisTrans(j,self.I)
            M2 = diagcat(M2,Mj)


        Lz = M2.T*M1.T*Lz*M1*M2
        Sz = M2.T*M1.T*Sz*M1*M2

        Hz = uB*B*(self.gL*Lz+self.gS*Sz)
        return Hz

    def CouplingMatrix(self, f, mf, q, Delta, n=2, B=0., I=1.):

        ng = 2
        lg = 0.0
        jg = 0.5

        H, F, Fz, J = self.FS(ng, lg)
        Hz = self.M1int(lg, B=B)
        eVal, eVec = eigh(H + Hz)
        egn = eVal
        Mg = np.matrix(eVec)
        fgn = np.diag(F)
        mfgn = np.diag(Fz)

        ks = np.asarray(np.argmax(np.abs(Mg), axis=0)).squeeze()

        i, = np.where((fgn == f) & (mfgn == mf))
        i, = np.where(i==ks)
        eg = egn[i]
        cgn = np.asarray(Mg[:, i]).squeeze()
        icgn = np.where(np.abs(cgn) > 1e-5)[0]

        le = 1.0
        H, F, Fz, J = self.FS(n, le)
        Hz = self.M1int(le, B=B)
        eVal, eVec = eigh(H + Hz)
        een = eVal
        Me = np.matrix(eVec)
        fen = np.diag(F)
        mfen = np.diag(Fz)
        jen = np.diag(J)

        AC = np.zeros(len(Delta))
        Pe = np.zeros(len(Delta))

        Matrix = np.zeros((Me.shape[0], Mg.shape[0]))

        for ig in icgn:
            cg = cgn[ig]
            fg = fgn[ig]
            mfg = mfgn[ig]

            for j in range(len(een)):
                ee = een[j]
                cen = np.asarray(Me[:, j]).squeeze()
                icen = np.where(np.abs(cen)>0)[0]

                for ie in icen:
                    ce = cen[ie]
                    mfe = mfen[ie]
                    fe = fen[ie]
                    je = jen[ie]

                    rme_j = np.sqrt(2*je+1) / np.sqrt(2)
                    # rme_j = 1

                    omg = (cg * ce * np.sqrt(I) * rme_j
                             * self.getSphericalDipoleMatrixElement(fg, mfg, fe, mfe, q)
                             * self._reducedMatrixElementFJ(jg, fg, je, fe)
                             )

                    Matrix[ie,ig] = omg

        return (Matrix, fen, mfen, fgn, mfgn)
    # def SingleAddress(self, f, mf, q, Delta, n=2, B=0., I=1.):
    #
    #     ng = 2
    #     lg = 0.0
    #     jg = 0.5
    #
    #     H, F, Fz, J = self.FS(ng, lg)
    #     Hz = self.M1int(lg, B=B)
    #     eVal, eVec = eigh(H + Hz)
    #     egn = eVal
    #     Mg = np.matrix(eVec)
    #     fgn = np.diag(F)
    #     mfgn = np.diag(Fz)
    #
    #     ks = np.asarray(np.argmax(np.abs(Mg), axis=0)).squeeze()
    #
    #     i, = np.where((fgn == f) & (mfgn == mf))
    #     i, = np.where(i==ks)
    #     eg = egn[i]
    #     cgn = np.asarray(Mg[:, i]).squeeze()
    #     icgn = np.where(np.abs(cgn) > 1e-5)[0]
    #
    #     le = 1.0
    #     H, F, Fz, J = self.FS(n, le)
    #     Hz = self.M1int(le, B=B)
    #     eVal, eVec = eigh(H + Hz)
    #     een = eVal
    #     Me = np.matrix(eVec)
    #     fen = np.diag(F)
    #     mfen = np.diag(Fz)
    #     jen = np.diag(J)
    #
    #     AC = np.zeros(len(Delta))
    #     Pe = np.zeros(len(Delta))
    #
    #     for ig in icgn:
    #         cg = cgn[ig]
    #         fg = fgn[ig]
    #         mfg = mfgn[ig]
    #
    #         for j in range(len(een)):
    #             ee = een[j]
    #             cen = np.asarray(Me[:, j]).squeeze()
    #             icen = np.where(np.abs(cen)>1e-15)[0]
    #
    #             for ie in icen:
    #                 ce = cen[ie]
    #                 mfe = mfen[ie]
    #                 fe = fen[ie]
    #                 je = jen[ie]
    #
    #                 rme_j = np.sqrt(2*je+1) / np.sqrt(2)
    #                 # rme_j = 1
    #
    #                 omg = (cg * ce * np.sqrt(I) * rme_j
    #                          * self.getSphericalDipoleMatrixElement(fg, mfg, fe, mfe, q)
    #                          * self._reducedMatrixElementFJ(jg, fg, je, fe)
    #                          )
    #
    #                 AC += omg ** 2 / (4 * (Delta - ee + eg))
    #
    #                 # Two-Photon Rabi Frequency
    #                 # OmegaR += Omaf0 * Ombf1 / (2 * (Delta - Ehfs))*0
    #
    #                 # Excitated state population Pe
    #                 Pe += omg ** 2 / (4 * (Delta - ee + eg) ** 2)
    #
    #     res = {
    #         "AC": AC,
    #         "Pe": Pe
    #     }
    #     return res
    def SingleAddress(self, f, mf, q, Delta, n=2, B=0., I=1.):

        ng = 2
        lg = 0.0
        jg = 0.5

        H, F, Fz, J = self.FS(ng, lg)
        Hz = self.M1int(lg, B=50e-4)
        eVal, eVec = eigh(H + Hz)
        egn = eVal
        Mg = np.matrix(eVec)
        fgn = np.diag(F)
        mfgn = np.diag(Fz)

        ks = np.asarray(np.argmax(np.abs(Mg), axis=0)).squeeze()

        i, = np.where((fgn == f) & (mfgn == mf))
        i, = np.where(i==ks)

        Hz = self.M1int(lg, B=B)
        eVal, eVec = eigh(H + Hz)
        egn = eVal
        Mg = np.matrix(eVec)

        eg = egn[i]
        cgn = np.asarray(Mg[:, i]).squeeze()
        icgn = np.argsort(np.abs(cgn))
        icgn = icgn[-2:]

        le = 1.0
        H, F, Fz, J = self.FS(n, le)
        Hz = self.M1int(le, B=B)
        eVal, eVec = eigh(H + Hz)
        een = eVal
        Me = np.matrix(eVec)
        fen = np.diag(F)
        mfen = np.diag(Fz)
        jen = np.diag(J)

        AC = np.zeros(len(Delta))
        Pe = np.zeros(len(Delta))

        for j in range(len(een)):
            ee = een[j]
            cen = np.asarray(Me[:, j]).squeeze()
            icen = np.arange(len(cen))
            omg = 0

            for ig in icgn:
                cg = cgn[ig]
                fg = fgn[ig]
                mfg = mfgn[ig]

                for ie in icen:
                    ce = cen[ie]
                    mfe = mfen[ie]
                    fe = fen[ie]
                    je = jen[ie]

                    rme_j = np.sqrt(2*je+1) / np.sqrt(2)
                    # rme_j = 1

                    omg += (cg * ce * np.sqrt(I) * rme_j
                             * self.getSphericalDipoleMatrixElement(fg, mfg, fe, mfe, q)
                             * self._reducedMatrixElementFJ(jg, fg, je, fe)
                             )

            AC += omg ** 2 / (4 * (Delta - ee + eg))

                    # Two-Photon Rabi Frequency
                    # OmegaR += Omaf0 * Ombf1 / (2 * (Delta - Ehfs))*0

                    # Excitated state population Pe
            Pe += omg ** 2 / (4 * (Delta - ee + eg) ** 2)

        res = {
            "AC": AC,
            "Pe": Pe
        }
        return res

    def SingleAddressFull(self, q, Delta, n=2, B=0., I=1.):

        ng = 2
        lg = 0.0
        jg = 0.5

        H, F, Fz, J = self.FS(ng, lg)
        Hz = self.M1int(lg, B=B)
        eVal, eVec = eigh(H + Hz)
        egn = eVal
        Mg = np.matrix(eVec)
        fgn = np.diag(F)
        mfgn = np.diag(Fz)

        le = 1.0
        H, F, Fz, J = self.FS(n, le)
        Hz = self.M1int(le, B=B)
        eVal, eVec = eigh(H + Hz)
        een = eVal
        Me = np.matrix(eVec)
        fen = np.diag(F)
        mfen = np.diag(Fz)
        jen = np.diag(J)

        AC = np.zeros(len(egn))
        Pe = np.zeros(len(egn))

        for i in range(len(egn)):
            eg = egn[i]
            cgn = np.asarray(Mg[:, i]).squeeze()
            icgn = np.argsort(np.abs(cgn))
            icgn = icgn[-2:]

            for j in range(len(een)):
                ee = een[j]
                cen = np.asarray(Me[:, j]).squeeze()
                icen = np.arange(len(cen))
                omg = 0

                for ig in icgn:
                    cg = cgn[ig]
                    fg = fgn[ig]
                    mfg = mfgn[ig]

                    for ie in icen:
                        ce = cen[ie]
                        mfe = mfen[ie]
                        fe = fen[ie]
                        je = jen[ie]

                        rme_j = np.sqrt(2*je+1) / np.sqrt(2)
                        # rme_j = 1

                        omg += (cg * ce * np.sqrt(I) * rme_j
                                 * self.getSphericalDipoleMatrixElement(fg, mfg, fe, mfe, q)
                                 * self._reducedMatrixElementFJ(jg, fg, je, fe)
                                 )

            AC[i] += omg ** 2 / (4 * (Delta - ee + eg))

                    # Two-Photon Rabi Frequency
                    # OmegaR += Omaf0 * Ombf1 / (2 * (Delta - Ehfs))*0

                    # Excitated state population Pe
            Pe[i] += omg ** 2 / (4 * (Delta - ee + eg) ** 2)

        res = {
            "egn": egn,
            "een": een,
            "AC": AC,
            "Pe": Pe
        }
        return res

    # def RamanAddress(self, f1, mf1, f2, mf2, qa, qb, Delta, n=2, B=0., Ia=1., Ib=1.):
    #
    #     ng = 2
    #     lg = 0.0
    #     jg = 0.5
    #
    #     H, F, Fz, J = self.FS(ng, lg)
    #     Hz = self.M1int(lg, B=B)
    #     eVal, eVec = eigh(H + Hz)
    #     egn = eVal
    #     Mg = np.matrix(eVec)
    #     fgn = np.diag(F)
    #     mfgn = np.diag(Fz)
    #
    #     ks = np.asarray(np.argmax(np.abs(Mg), axis=0)).squeeze()
    #
    #     # fgn = fgn[ks]
    #     # mfgn = mfgn[ks]
    #
    #     i1, = np.where((fgn == f1) & (mfgn == mf1))
    #     i2, = np.where((fgn == f2) & (mfgn == mf2))
    #     i1, = np.where(i1==ks)
    #     i2, = np.where(i2==ks)
    #     eg1 = egn[i1]
    #     eg2 = egn[i2]
    #     cg1n = np.asarray(Mg[:, i1]).squeeze()
    #     cg2n = np.asarray(Mg[:, i2]).squeeze()
    #     # icg1n = np.where(np.abs(cg1n) > 1e-2)[0]
    #     icg1n = np.argsort(np.abs(cg1n))
    #     icg1n = icg1n[-2:]
    #     # icg1n = np.arange(len(cg1n))
    #     # icg2n = np.where(np.abs(cg2n) > 1e-2)[0]
    #     icg2n = np.argsort(np.abs(cg2n))
    #     icg2n = icg2n[-2:]
    #     # icg2n = np.arange(len(cg2n))
    #     omega = eg2 - eg1
    #     Deltaa = Delta - eg1
    #     Deltab = Delta - eg2
    #
    #     le = 1.0
    #     H, F, Fz, J = self.FS(n, le)
    #     Hz = self.M1int(le, B=B)
    #     eVal, eVec = eigh(H + Hz)
    #     een = eVal
    #     Me = np.matrix(eVec)
    #     fen = np.diag(F)
    #     mfen = np.diag(Fz)
    #     jen = np.diag(J)
    #
    #     AC1 = np.zeros(len(Delta))
    #     AC2 = np.zeros(len(Delta))
    #     OmegaR = np.zeros(len(Delta))
    #     Pe = np.zeros(len(Delta))
    #
    #     T1 = []
    #
    #     for ig1 in icg1n:
    #         for ig2 in icg2n:
    #             cg1 = cg1n[ig1]
    #             cg2 = cg2n[ig2]
    #             fg1 = fgn[ig1]
    #             fg2 = fgn[ig2]
    #             mfg1 = mfgn[ig1]
    #             mfg2 = mfgn[ig2]
    #
    #             for j in range(len(een)):
    #                 ee = een[j]
    #                 cen = np.asarray(Me[:, j]).squeeze()
    #                 icen = np.where(np.abs(cen)>1e-3)[0]
    #                 # icen = np.arange(len(cen))
    #                 # icen = [np.argmax(np.abs(cen))]
    #
    #                 for ie in icen:
    #                     ce = cen[ie]
    #                     mfe = mfen[ie]
    #                     fe = fen[ie]
    #                     je = jen[ie]
    #
    #                     rme_j = np.sqrt(2*je+1) / np.sqrt(2)
    #                     # rme_j = 1
    #
    #                     omga1 = (cg1 * ce * np.sqrt(Ia) * rme_j
    #                              * self.getSphericalDipoleMatrixElement(fg1, mfg1, fe, mfe, qa)
    #                              * self._reducedMatrixElementFJ(jg, fg1, je, fe)
    #                              )
    #                     omga2 = (cg2 * ce * np.sqrt(Ia) * rme_j
    #                              * self.getSphericalDipoleMatrixElement(fg2, mfg2, fe, mfe, qa)
    #                              * self._reducedMatrixElementFJ(jg, fg2, je, fe)
    #                              )
    #                     omgb1 = (cg1 * ce * np.sqrt(Ib) * rme_j
    #                              * self.getSphericalDipoleMatrixElement(fg1, mfg1, fe, mfe, qb)
    #                              * self._reducedMatrixElementFJ(jg, fg1, je, fe)
    #                              )
    #                     omgb2 = (cg2 * ce * np.sqrt(Ib) * rme_j
    #                              * self.getSphericalDipoleMatrixElement(fg2, mfg2, fe, mfe, qb)
    #                              * self._reducedMatrixElementFJ(jg, fg2, je, fe)
    #                              )
    #                     AC1 += (omga1 ** 2 / (4 * (Deltaa - ee))
    #                             + omgb1 ** 2 / (4 * (Deltab - ee)))
    #                     AC2 += (omga2 ** 2 / (4 * (Deltaa - ee))
    #                             + omgb2 ** 2 / (4 * (Deltab - ee)))
    #
    #                     OmegaR += omga1 * omgb2 / (2*(Deltaa - ee))
    #
    #                     Pe += (omga1 ** 2 / (4 * (Deltaa - ee)**2)
    #                             # + omgb1 ** 2 / (4 * (Deltab - ee)**2)
    #                             # + omga2 ** 2 / (4 * (Deltaa - ee)**2)
    #                             + omgb2 ** 2 / (4 * (Deltab - ee)**2)
    #                            )/2
    #
    #
    #
    #     res = {
    #         "AC1": AC1,
    #         "AC2": AC2,
    #         "OmegaR": OmegaR,
    #         "Pe": Pe,
    #     }
    #     return res

    def RamanAddress(self, f1, mf1, f2, mf2, qa, qb, Delta, n=2, B=0., Ia=1., Ib=1.):

        ng = 2
        lg = 0.0
        jg = 0.5

        H, F, Fz, J = self.FS(ng, lg)
        Hz = self.M1int(lg, B=10e-4)
        eVal, eVec = eigh(H + Hz)
        egn = eVal
        Mg = np.matrix(eVec)
        fgn = np.diag(F)
        mfgn = np.diag(Fz)

        ks = np.asarray(np.argmax(np.abs(Mg), axis=0)).squeeze()

        i1, = np.where((fgn == f1) & (mfgn == mf1))
        i2, = np.where((fgn == f2) & (mfgn == mf2))
        i1, = np.where(i1==ks)
        i2, = np.where(i2==ks)

        Hz = self.M1int(lg, B=B)
        eVal, eVec = eigh(H + Hz)
        egn = eVal
        Mg = np.matrix(eVec)

        eg1 = egn[i1]
        eg2 = egn[i2]
        cg1n = np.asarray(Mg[:, i1]).squeeze()
        cg2n = np.asarray(Mg[:, i2]).squeeze()
        # icg1n = np.where(np.abs(cg1n) > 1e-2)[0]
        icg1n = np.argsort(np.abs(cg1n))
        icg1n = icg1n[-3:]
        # icg1n = np.arange(len(cg1n))
        # icg2n = np.where(np.abs(cg2n) > 1e-2)[0]
        icg2n = np.argsort(np.abs(cg2n))
        icg2n = icg2n[-3:]
        # icg2n = np.arange(len(cg2n))
        omega = eg2 - eg1
        Deltaa = Delta - eg1
        Deltab = Delta - eg2

        le = 1.0
        H, F, Fz, J = self.FS(n, le)
        Hz = self.M1int(le, B=B)
        eVal, eVec = eigh(H + Hz)
        een = eVal
        Me = np.matrix(eVec)
        fen = np.diag(F)
        mfen = np.diag(Fz)
        jen = np.diag(J)

        AC1 = np.zeros(len(Delta))
        AC2 = np.zeros(len(Delta))
        OmegaR = np.zeros(len(Delta))
        Pe = np.zeros(len(Delta))

        T1 = []

        for j in range(len(een)):
            ee = een[j]
            cen = np.asarray(Me[:, j]).squeeze()
            icen = np.arange(len(cen))
            omga1 = 0
            omgb1 = 0
            omga2 = 0
            omgb2 = 0
            for ig1 in icg1n:
                for ig2 in icg2n:
                    cg1 = cg1n[ig1]
                    cg2 = cg2n[ig2]
                    fg1 = fgn[ig1]
                    fg2 = fgn[ig2]
                    mfg1 = mfgn[ig1]
                    mfg2 = mfgn[ig2]

                    for ie in icen:
                        ce = cen[ie]
                        mfe = mfen[ie]
                        fe = fen[ie]
                        je = jen[ie]
                        rme_j = np.sqrt(2 * je + 1) / np.sqrt(2)

                        omga1 += (cg1 * ce * np.sqrt(Ia) * rme_j
                                 * self.getSphericalDipoleMatrixElement(fg1, mfg1, fe, mfe, qa)
                                 * self._reducedMatrixElementFJ(jg, fg1, je, fe)
                                 )
                        omga2 += (cg2 * ce * np.sqrt(Ia) * rme_j
                                 * self.getSphericalDipoleMatrixElement(fg2, mfg2, fe, mfe, qa)
                                 * self._reducedMatrixElementFJ(jg, fg2, je, fe)
                                 )
                        omgb1 += (cg1 * ce * np.sqrt(Ib) * rme_j
                                 * self.getSphericalDipoleMatrixElement(fg1, mfg1, fe, mfe, qb)
                                 * self._reducedMatrixElementFJ(jg, fg1, je, fe)
                                 )
                        omgb2 += (cg2 * ce * np.sqrt(Ib) * rme_j
                                 * self.getSphericalDipoleMatrixElement(fg2, mfg2, fe, mfe, qb)
                                 * self._reducedMatrixElementFJ(jg, fg2, je, fe)
                                 )
            AC1 += (omga1 ** 2 / (4 * (Deltaa - ee))
                    + omgb1 ** 2 / (4 * (Deltab - ee)))
            AC2 += (omga2 ** 2 / (4 * (Deltaa - ee))
                    + omgb2 ** 2 / (4 * (Deltab - ee)))

            OmegaR += omga1 * omgb2 / (2*(Deltaa - ee))

            Pe += (omga1 ** 2 / (4 * (Deltaa - ee)**2)
                    + omgb1 ** 2 / (4 * (Deltab - ee)**2)
                    + omga2 ** 2 / (4 * (Deltaa - ee)**2)
                    + omgb2 ** 2 / (4 * (Deltab - ee)**2)
                   )/2



        res = {
            "AC1": AC1,
            "AC2": AC2,
            "OmegaR": OmegaR,
            "Pe": Pe,
            "Deltaa": Deltaa,
            "Deltab": Deltab
        }
        return res

    def ReducedMatrixElement(self, n):

        return np.sqrt(self.Gamma[n - 2] * 3 * np.pi * epsilon_0 * hbar * (self.wavelen[n - 2] / 2 / np.pi) ** 3)

    def RabiFreq(self, n, laserPower, waist=False,NA=0.6):
        if not waist:
            laserWaist = self.wavelen[n - 2] / np.pi / NA * 2
        else:
            laserWaist = waist

        maxIntensity = 2 * laserPower / (np.pi * laserWaist ** 2)
        electricField = np.sqrt(2.0 * maxIntensity / (3e8 * epsilon_0))
        dipole = self.ReducedMatrixElement(n)

        freq = electricField * abs(dipole) / hbar / 2 / np.pi
        return freq

    def RabiFreqIs(self, n, Is):
        I = 2.54 * Is * 10
        electricField = np.sqrt(2.0 * I / (3e8 * epsilon_0))
        dipole = self.ReducedMatrixElement(n)

        freq = electricField * abs(dipole) / hbar / 2 / pi
        return freq


