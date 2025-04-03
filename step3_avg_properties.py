# %% [markdown]
# The properties retrieved for sample/P/T points can now be queried to calculate things like
# the average density or Mg# throughout the velocity model. If you also want to go back and look
# at pressure and temperature you'll have to get those using the procedure in step1.

# %%
# import libraries
import duckdb
import numpy as np
import pandas as pd
import utils as ut
import os, sys

# %% 
# connect db/tables
q0 = 80  # surface heat flow
conn = duckdb.connect("example_data/%i-output.db" % q0, read_only=True)

# %%
# get all the x/z points in the model
xzdf = conn.sql("SELECT ix, iz FROM matches GROUP BY ix, iz").df()

# %%
# loop points
mgn = np.zeros((21,441))*np.nan  # same shape as the whole input velocity model
for i, row in xzdf.iterrows():
    samps = conn.sql("SELECT sample_id, xz_ip, xz_it FROM matches WHERE ix=%i AND iz=%i" % (row.ix, row.iz)).df()
    samps = samps.rename(mapper={"sample_id":"id","xz_ip":"ip","xz_it":"it"},axis=1)
    this_point = conn.sql("SELECT * FROM sample NATURAL JOIN samps").df()
    # and here you can do whatever math you want with the results
    avgs = this_point[['wtSiO2','Mgnum','rho']].mean()  # for example, average numerical quantities
    mgn[row.ix,row.iz] = avgs['Mgnum']  # and save those values in an ordered way

conn.close()

# %% [markdown]
# From here, you might want to plot averaged quantities, write them out, or dig more into
# spatial/depth patterns. Note that the `mgn` array containing the averaged Mg# values will have
# a lot of empty space in it for portions of the model outside of the P/T range of the database.
