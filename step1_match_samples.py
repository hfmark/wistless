# %% [markdown]
# This script takes a 2D Vp model, estimates P and T based for a conductive
# geotherm based on surface heat flow, and queries for each P/T/Vp point to
# find matching samples in the database. Sample IDs for matches are saved in
# another db for later.

# %% [markdown]
# This is not a particularly fast process (though using duckdb speeds things up 
# significantly compared to other database options, we promise). The example
# velocity model here is therefore small. For a real calculation with a lot more
# velocities/queries, we recommend using HPC. duckdb will parallelize reads automatically,
# but it will only write with one thread so there's a point where more processes don't
# really help anymore.
#
# In practice, for a velocity model with ~900 points in X and ~100 in Z within the db P/T range,
# we've found that on a shared cluster, n=8 processes running for 24 hours works most of the time.
# Depending on the velocities and the geotherm you will have more or fewer matches to each
# query; scenarios with more matches will take longer to run.

# %%
# import libraries
import duckdb
import numpy as np
import pandas as pd
import utils as ut
from scipy.interpolate import interp1d

# %%
# read in a velocity model and calculate depth below seafloor
vmod = ut.vmodel()
vmod.read_vm_from_file('example_data/example_Vp_model.vm')
vmod.apply_jumps_to_grid()
vels = 1./vmod.slown_complete  # invert slowness to Vp

seaf = vmod.layer[0]
zvals = np.tile(np.linspace(vmod.z1,vmod.z2,vmod.nz),(vmod.nx,1))
for i in range(vmod.nx):
    zvals[i,:] = zvals[i,:] - seaf[i]

# %%
# make a lookup table for P and T with a specified crustal thickness and surface heat flow
cont = ut.continent()
cont.CM = 28  # moho depth for setting layer thicknesses in P(z) calculation
q0 = 80  # surface heat flow, mW/m^2
zpt,Tpt,Apt,qpt,kpt = cont.calc_T_profile(q0*1e-3,zb=vmod.z2)
Ppt = [cont.z_to_P(e) for e in zpt]

# %%
# look up and interpolate P and T across velocity model
pressure_interp = interp1d(zpt,Ppt,fill_value='extrapolate')
temperat_interp = interp1d(zpt,Tpt,fill_value='extrapolate')

pres = np.zeros(vels.shape)
temp = np.zeros(vels.shape)
for i in range(vmod.nz):
    pres[:,i] = pressure_interp(zvals[:,i])
    temp[:,i] = temperat_interp(zvals[:,i])
temp = temp - 273.15  # convert K to C

# %%
# connect to db, retrieve some info
dbname = "hacker_abqtz"
conn = duckdb.connect("Data/%s.db" % dbname)  # the wistless database
ptLH = conn.sql("SELECT min(pressure), max(pressure), min(temperature), max(temperature) FROM %s.pt" % dbname).fetchall()[0]
minP,maxP,minT,maxT = ptLH
dtype = dict(conn.sql("select column_name, data_type from information_schema.columns").fetchall())

# %%
# attach a second db for outputs and make a table there
conn.sql("ATTACH 'example_data/%i-output.db' AS output" % (q0))  # output db for results

# %% [markdown]
# NOTE this next line will overwrite the 'matches' table in the output file if it exists already.
# If you want to start a new table, you can either create a new db (change path in the "ATTACH" line above) or create a new table in the existing db (change the name 'matches' throughout the script).
# To append lines to an existing 'matches' table, just comment out the "CREATE OR REPLACE TABLE" line below.

# %%
conn.sql("CREATE OR REPLACE TABLE output.matches (ix INTEGER, iz INTEGER, sample_id INTEGER, xz_ip INTEGER, xz_it INTEGER);")
cursor = conn.cursor()

# %%
# loop!
ret = ['id','ip','it']  # columns to return from query
for ix in range(vmod.nx):
    print(ix)  # quasi-progress bar
    for iz in range(vmod.nz):
        # CHECK if p/t are even in range - we can only really do lower crust
        if pres[ix,iz] < minP or pres[ix,iz] > maxP or temp[ix,iz] < minT or temp[ix,iz] > maxT:
            continue
        to_filter = ['pressure','temperature','vp']  # filter on these columns
        rads = ['=','+-','%']  # match P (=), T in +/- range, Vp within a % range
        vals = [pres[ix,iz],[temp[ix,iz],5],[vels[ix,iz],0.5]]  # note T spacing in db is 10*

        q1 = ut.construct_query_new(cursor,to_filter,rads,vals,ret,dtype,ptLH)
        df = conn.sql(q1).df()

        if len(df) != 0:  # if query returned nonzero lines from db
            for uid in df.id.unique():  # add each unique returned sample to 'matches'
                conn.sql("INSERT INTO output.matches BY POSITION VALUES (%i, %i, %i, %i, %i);" % (ix, iz, uid, df.ip.unique()[0],df.it.unique()[0]))  # save sample, P, T, and model x/z location
        conn.commit()

conn.close()

