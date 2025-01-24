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

def read_markdown_file(mdfile):
    return Path(mdfile).read_text()


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
        elif dtypes[to_filter[i]] == "VARCHAR":  # rads can only be = or !=
            ands.append("%s %s '%s'" % (to_filter[i],rads[i],vals[i]))
    q1 = "SELECT %s FROM hacker_all.arr WHERE " % (', '.join(ret))
    for i,a in enumerate(ands):
        q1 += a
        if i != len(ands)-1:
            q1 += " AND "

    return q1
    
