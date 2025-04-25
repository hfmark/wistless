# %% [markdown]
# Averages are one way to look at the results of queries; there are other options too.
# This script demonstrates joint misfit calculations, fitting, and misfit-weighted averaging
# for a database query. This example is independent of the other Python example scripts - it
# is not based on the example velocity model.

# %% 
# import libraries
import duckdb
import numpy as np
import pandas as pd
import utils as ut
import os, sys

# %% 
# connect to the sample database
dbname = "hacker_abqtz"
conn = duckdb.connect("Data/%s.db" % dbname)  # the wistless database
cursor = conn.cursor()
ptLH = conn.sql("SELECT min(pressure), max(pressure), min(temperature), max(temperature) FROM %s.pt" % dbname).fetchall()[0]
dtype = dict(conn.sql("select column_name, data_type from information_schema.columns").fetchall())


# %%
# set up a query that filters on at least one of vp, vs, and vp/vs ratio
# here we will us an arbitrary query that we know returns a reasonable number of matching points
to_filter = ['vp','vs']
rads = ['+-','+-']
vals = [[8.,0.1],[5.,.1]]
ret = ['vp','vs','rho','temperature','pressure','Al2O3']  # return vp, vs, T, and some other things

# %%
# construct and run the query
q1 = ut.construct_query_new(cursor,to_filter,rads,vals,ret,dtype,ptLH)
df = conn.sql(q1).df()  # pd.DataFrame with results


# %%
# calculate misfit with respect to vp and vs filters, and joint misfit -> add as df columns
df = ut.joint_misfit(df,to_filter,vals,rads)

# %%
# find the best-fit temperature for this set of sample
# method 1: minimize joint misfit
# method 2: gaussian weighting of temperatures for matching points
best_T_misfit = df.iloc[df['joint_misfit'].argmin()]['temperature']
best_T_gauss = ut.gaussian_best_T(df)
print('misfit T: ',best_T_misfit)
print('gaussian T: ',best_T_gauss)

# %%
# use joint misfit to calculate a weighted mean and standard deviation for other db properties
# in this case, density and wt% Al2O3
# this calculation is done at a specific P/T point, and requires that the query we are working
# with returned enough matching samples at that P/T point to have meaningful statistics.
# you could also adjust this to calculate for a specific P/T range, or for all matched samples.

# the P/T point, approximately:
pfit = 0.5  # GPa
tfit = 450  # *C

# find the matches from this query for this P/T point
pr_unique = np.unique(df['pressure'].values)
tm_unique = np.unique(df['temperature'].values)
pfit_actual = pr_unique[np.argmin(abs(pr_unique - pfit))]
tfit_actual = tm_unique[np.argmin(abs(tm_unique - tfit))]
sel = df[(df['pressure'] == pfit_actual)&(df['temperature'] == tfit_actual)]

# do some checks
assert 'joint_misfit' in df.columns, 'joint misfit must be calculated'
assert len(sel) > 1, 'not enough samples at this P/T point'  # you may want a higher threshold

# calculate weighted mean and stdev
rho_mean = np.average(sel['rho'],weights=1./sel['joint_misfit'])
rho_stdv = np.sqrt(np.cov(sel['rho'], aweights=1./sel['joint_misfit']))
print('rho mean/std: ', rho_mean, rho_stdv)
al_mean = np.average(sel['Al2O3'],weights=1./sel['joint_misfit'])
al_stdv = np.sqrt(np.cov(sel['Al2O3'], aweights=1./sel['joint_misfit']))
print('Al2O3 mean/std: ', al_mean, al_stdv)
