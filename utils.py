import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import os, sys


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
        qp = "SELECT id FROM hacker_noamph.pt WHERE %s BETWEEN ? AND ? ORDER BY abs(?-%s) LIMIT 1" % (col[pt],col[pt])
        ipt = cursor.execute(qp,(low,hgh,tofit[0])).fetchall()[0][0]
        ands = "%s = %i " % (ii[pt],ipt)
    elif len(tofit) == 2:
        ands = "%s IN (SELECT id FROM hacker_noamph.pt WHERE %s BETWEEN %f AND %f)" % (ii[pt],col[pt],tofit[0],tofit[1])
        qp = "SELECT id FROM hacker_noamph.pt WHERE %s BETWEEN ? AND ?" % (col[pt])
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
