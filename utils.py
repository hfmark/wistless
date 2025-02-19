import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from array import array
from copy import copy
import os, sys

########################################################################
# duckdb and queries
########################################################################

def pt_select(cursor,pt,tofit,return_and=True):
    """ get ip or it values 

    if return_and, return the condition for sql query; if not, return the ip/it values
    """
    dp = {'p': 0.1, 't': 10}
    col = {'p':'pressure','t':'temperature'}
    ii = {'p':'ip','t':'it'}
    if not hasattr(tofit,'__len__'):
            tofit = np.atleast_1d(tofit)
    if len(tofit) == 2 and tofit[0] == tofit[1]:
        tofit = [tofit[0],]
    if len(tofit) == 1:
        low = tofit[0] - dp[pt]; hgh = tofit[0] + dp[pt]
        qp = "SELECT id FROM hacker_all.pt WHERE %s BETWEEN ? AND ? ORDER BY abs(?-%s) LIMIT 1" % (col[pt],col[pt])
        ipt = cursor.execute(qp,(low,hgh,tofit[0])).fetchall()[0][0]
        ands = "%s = %i " % (ii[pt],ipt)
    elif len(tofit) == 2:
        ands = "%s IN (SELECT id FROM hacker_all.pt WHERE %s BETWEEN %f AND %f)" % (ii[pt],col[pt],tofit[0],tofit[1])
        qp = "SELECT id FROM hacker_all.pt WHERE %s BETWEEN ? AND ?" % (col[pt])
        _ = cursor.execute(qp,(tofit[0],tofit[1]))
        ipt = np.array([e[0] for e in cursor.fetchall()])
    if return_and:
        return ands
    else:
        return ipt


def joint_misfit(df,filts,vals,rads):
    """ Calculate mistfits and joint misfit for a set of points and filter conditons

    df is a dataframe of points, from filtering database, that has vp, vs, and/or vpvs
    filts is the list of quantities used to filter (inc vp, vs, and/or vpvs)
    vals is the list of filter parameters (numerical)
    rads is the list of filter conditions (vp, vs, and/or vpvs with particular ones)
    """

    for i in range(len(rads)):
        if filts[i] in ['vp','vs','vpvs'] and rads[i] in ['in','=','+-','%'] and len(df) > 0 and filts[i] in df.columns:
            if rads[i] == 'in':
                fitval = vals[i][0] + (vals[i][1] - vals[i][0])/2
            else:
                fitval = vals[i][0]

            df['misfit_%s' % filts[i]] = (df[filts[i]] - fitval)/fitval

    joint_misfit = np.zeros(len(df))
    for col in df.columns:
        if col.startswith('misfit_'):
            joint_misfit += df[col]**2
    df['joint_misfit'] = np.sqrt(joint_misfit)
    return df

def gaussian_best_T(df):
    """ Gaussian best fit temperature from a set of points
    """

    Ts, counts = np.unique(df['temperature'],return_counts=True)
    sum_fits = sum(counts)
    sum_M2 = sum(counts*Ts**2)
    best_T = sum(Ts*counts)/sum_fits
    return best_T

def runquery(cur,q1):
    cur.execute(q1)
    df = cur.fetch_df()
    return df

def convert_df(df):
    return df.to_csv().encode("utf-8")

def construct_query_new(cur,to_filter,rads,vals,ret,dtypes,ptLH):
    """ build and-ed query to return certain fields
    structured so it works with the streamlit gui for inputs
    """
    p_lo,p_hi,t_lo,t_hi = ptLH
    ands = []
    for i in range(len(to_filter)):
        if dtypes[to_filter[i]] != "VARCHAR" and to_filter[i] not in ['pressure','temperature']:
            if rads[i] in ['<',">",'<=','>=','=','!=']:
                ands.append("%s %s %.2f" % (to_filter[i],rads[i].lstrip('\\'),vals[i]))
                if rads[i] in ['<','<=']:
                    ands.append("%s %s 0" % (to_filter[i],r'>='))  # screen out -999s that are nulls
            elif rads[i] == '+-':
                low = vals[i][0] - vals[i][1]
                hgh = vals[i][0] + vals[i][1]
                ands.append("%s between %f and %f" % (to_filter[i],low,hgh))
            elif rads[i] == '%':
                low = vals[i][0] - vals[i][1]*vals[i][0]/100
                hgh = vals[i][0] + vals[i][1]*vals[i][0]/100
                ands.append("%s between %f and %f" % (to_filter[i],low,hgh))
            elif rads[i] == 'in':  # not an option for strings, and it's slider only right now
                ands.append("%s between %f and %f" % (to_filter[i],vals[i][0],vals[i][1]))
        elif dtypes[to_filter[i]] != "VARCHAR" and to_filter[i] in ['pressure','temperature']:
            # figure out p_lo and p_hi basically
            if to_filter[i] == 'pressure': ptflag = 'p'; lolo = p_lo; hihi = p_hi
            if to_filter[i] == 'temperature': ptflag = 't'; lolo = t_lo; hihi = t_hi
            if rads[i] in ['<', "<="]:
                ands.append(pt_select(cur,ptflag,[lolo,vals[i]],return_and=True))
            elif rads[i] in ['>', ">="]:
                ands.append(pt_select(cur,ptflag,[vals[i],hihi],return_and=True))
            elif rads[i] == '=':
                ands.append(pt_select(cur,ptflag,[vals[i],],return_and=True))
            elif rads[i] == '+-':
                low = vals[i][0] - vals[i][1]
                hgh = vals[i][0] + vals[i][1]
                ands.append(pt_select(cur,ptflag,[low,hgh],return_and=True))
            elif rads[i] == '%':
                low = vals[i][0] - vals[i][1]*vals[i][0]/100
                hgh = vals[i][0] + vals[i][1]*vals[i][0]/100
                ands.append(pt_select(cur,ptflag,[low,hgh],return_and=True))
            elif rads[i] == 'in':  # this is the using-sliders case
                ands.append(pt_select(cur,ptflag,[vals[i][0],vals[i][1]],return_and=True))
        elif dtypes[to_filter[i]] == "VARCHAR":  # rads can only be = or !=
            ands.append("%s %s '%s'" % (to_filter[i],rads[i],vals[i]))
    q1 = "SELECT %s FROM hacker_all.arr WHERE " % (', '.join(ret))
    for i,a in enumerate(ands):
        q1 += a
        if i != len(ands)-1:
            q1 += " AND "

    return q1
    
########################################################################
# streamlit
########################################################################

def read_markdown_file(mdfile):
    return Path(mdfile).read_text()

########################################################################
# other stuff for examples
########################################################################

class continent:
    def __init__(self):
        # Furlong and Chapman 2013 (Pollack and Chapman 1977)
        self.k_up_cr = 3.0  # W/mK at RTP
        self.k_lw_cr = 2.6  # W/mK at RTP
        self.b_up_cr = 1.5e-3  # 1/K
        self.b_lw_cr = 1.0e-4  # 1/K
        self.c_ul_cr = 1.5e-3  # 1/km

        self.P = 0.4   # Pollack and Chapman 1977 (0.26 for Hasterok and Chapman 2011?)
        self.b = 2e4   # NOTE characteristic dimension for vertical heat production distributions???

        self.LC = 16  # km, depth to UC/LC switch
        self.CM = 30  # km, depth to LC/M switch

        self.Alc = 0.45*1e-6  # W/m^3, from F+C13
        self.Am = 0.02*1e-6   # W/m^3, from F+C13

        self.rho_uc = 2700  # kg/m^3
        self.rho_lc = 3000
        self.rho_mn = 3300

        self.g = 0.0098   # GPa/m

    def calc_k(self,T,z):
        """ z in km, since c is in 1/km
        T in K
        """
        T = T - 273.15  # *C, from K input
        c = self.c_ul_cr
        if z < self.LC:
            k0 = self.k_up_cr
            b = self.b_up_cr
            return k0*(1+c*z)/(1+b*T)  # W/mK bc k0 is W/mK
        elif z >= self.LC and z < self.CM:
            k0 = self.k_lw_cr
            b = self.b_lw_cr
            return k0*(1+c*z)/(1+b*T)  # W/mK bc k0 is W/mK
        else:
        # Calculation of Thermal Conductivity of the Mantle determined from T and z
        # Schatz, J. F., and G. Simmons (1972), Thermal conductivity of earth 
        # materials at high temperatures, Journal of Geophysical Research, 
        # 77(35), 6966–6983.
            # Eqn (9) pg 6975
            a = 7.4091778e-2
            b = 5.0191204e-4
            kl = 1.0/(a + b * T)
            # Eqn (10)
            kr = 0.0
            if T > 500.0:
                d = 2.3e-3 
                kr = d * (T-500.0)
            # Eqn (11)
            c = 1.255199
            klmin =  c * (1.0 + z /1e3)
            return klmin + max(kl, kr)

    def calc_A(self,z,A0):
        if z < self.LC:  # upper crust
            return A0*np.exp(-z/self.b)
        elif z>=self.LC and z < self.CM:  # lower crust
            return self.Alc
        else: # mantle
            return self.Am
    
    def calc_T_profile(self,q0,zb=300,T0=298.15):
        """
        q0 - surface heat flow, W/m^2
        T0 - surface temp, K
        zb - bottom depth for the profile, km
        """

        A0 = self.P*q0/self.b

        zpts = np.arange(0,zb,0.5)

        Tpts = np.zeros(zpts.shape)
        qpts = np.zeros(zpts.shape)
        Apts = np.zeros(zpts.shape)
        kpts = np.zeros(zpts.shape)
        Tpts[0] = T0; qpts[0] = q0; Apts[0] = A0
        kpts[0] = self.k_up_cr
        
        for i in range(1,len(Tpts)):
            dz = (zpts[i] - zpts[i-1])*1e3  # in meters!!
            kpts[i] = self.calc_k(Tpts[i-1],zpts[i-1])
            Apts[i] = self.calc_A(zpts[i],A0)
            qpts[i] = qpts[i-1] - Apts[i]*dz
            Tpts[i] = Tpts[i-1] + qpts[i-1]*dz/kpts[i] - (Apts[i]*dz**2)/(2*kpts[i])

        return zpts, Tpts, Apts, qpts, kpts

    def calc_adiabat(self,Tpot=1350,dTdz=0.4,zb=300):
        return [Tpot,Tpot+dTdz*zb], [0,zb]

    def z_to_P(self,z):
        """ z in km, P in GPa
        """
        if z < 0.1:
            return 0
        zpts = np.arange(0,z,0.1)
        rho = np.ones(zpts.shape)*self.rho_mn
        rho[zpts<self.CM] = self.rho_lc
        rho[zpts<self.LC] = self.rho_uc

        P = np.cumsum(rho*self.g*0.1/1e3)
        return P[-1]

class vmodel:
    """
    class for holding **2D** vm parameters: dimensions, grid, jumps, etc
    """

    def __init__(self):
        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.nr = 0

        self.dx = 0
        self.dy = 0
        self.dz = 0

        self.x1 = 0
        self.y1 = 0
        self.z1 = 0

        self.x2 = 0
        self.y2 = 0
        self.z2 = 0

        self.slown = np.array([])

        self.layer = np.array([])
        self.jump = np.array([])
        self.lind = np.array([])
        self.jind = np.array([])

        self.dim = 2  # this is maybe not necessary but a good reminder

    def read_vm_from_file(self,ifile):
        """
        read info from binary vm file, hold in this instance
        """
        assert os.path.exists(ifile), 'file does not exist'

        fin = open(ifile,'r+b')
        fin.seek(0)

        # first: dimensions and spacing
        nn = array('i')
        xyz = array('f')
        nn.fromfile(fin,4)
        xyz.fromfile(fin,9)

        self.nx = nn[0]; self.ny = nn[1]
        self.nz = nn[2]; self.nr = nn[3]

        self.x1 = xyz[0]; self.y1 = xyz[1]; self.z1 = xyz[2]
        self.x2 = xyz[3]; self.y2 = xyz[4]; self.z2 = xyz[5]
        self.dx = xyz[6]; self.dy = xyz[7]; self.dz = xyz[8]

        # check dim
        assert self.ny == 1, 'this is not a 2D file; read as 3D and slice instead'

        # second: background slowness grid
        arr = array('f')
        arr.fromfile(fin,self.nx*self.nz)
        self.slown = np.array(arr).reshape(self.nx,self.nz)

        # third: surfaces depths, jumps, and indices(x2)
        arr = array('f')
        arr.fromfile(fin,self.nr*self.nx)
        self.layer = np.array(arr).reshape(self.nr,self.nx)

        arr = array('f')
        arr.fromfile(fin,self.nr*self.nx)
        self.jump = np.array(arr).reshape(self.nr,self.nx)

        arr = array('i')
        arr.fromfile(fin,self.nr*self.nx)
        self.lind = np.array(arr).reshape(self.nr,self.nx)

        arr = array('i')
        arr.fromfile(fin,self.nr*self.nx)
        self.jind = np.array(arr).reshape(self.nr,self.nx)

        fin.close()

    def apply_jumps_to_grid(self):
        """
        add jumps to background slowness within layers
        """

        grid = copy(self.slown)

        # loop over interfaces
        for i in range(self.nr):
            # loop over x and y
            for ix in range(self.nx):
                # find layer depth in grid
                iz1 = int((self.layer[i,ix] - self.z1)/self.dz) + 1
                if i < self.nr-1:  # not the last layer yet
                    iz2 = int((self.layer[i+1,ix] - self.z1)/self.dz) + 1
                else:
                    iz2 = self.nz - 1
                #if self.jind[i,ix] != 0:
                    #grid[ix,iz1:iz2] += self.jump[i,ix]
                grid[ix,iz1:iz2] += self.jump[i,ix]
        self.slown_complete = grid
