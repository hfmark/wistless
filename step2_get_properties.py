# %% [markdown]
# The next step here is to take the set of unique sample/P/T rows that match points in our
# velocity model and query the database to get all the info on each sample.
#
# This is done separately from the initial query because listing all sample/P/T matches first
# enables us to look only for unique entries' full properties. The same sample/P/T point
# may match multiple x/z/Vp spots in the velocity model - in a reasonable model of Earth structure,
# it's very likely to have one sample/P/T point come up multiple times. For this example, the number
# of "matches" from step1 is 457,549 while the number of unique sample/P/T points in that set is 
# 16,396.

# %%
# import libraries
import duckdb
import numpy as np
import pandas as pd
import utils as ut

# %% 
# parameters we still need:
q0 = 80 # (this is in the database file names)

# %%
# database connections and attachments
dbname = "hacker_abqtz"
conn = duckdb.connect("Data/%s.db" % dbname) #,read_only=True)
conn.sql("ATTACH 'example_data/%i-output.db' AS output" % (q0))
ptLH = conn.sql("SELECT min(pressure), max(pressure), min(temperature), max(temperature) FROM %s.pt" % dbname).fetchall()[0]
minP,maxP,minT,maxT = ptLH
cursor = conn.cursor()
dtype = dict(conn.sql("select column_name, data_type from information_schema.columns").fetchall())

# %%
# loop outputs table: 
# get unique triplets of sample, ip, it
# query for their compositions -> save in another table, which we create at iteration 0
sids = conn.sql("SELECT DISTINCT sample_id FROM output.matches;").df()
for j,sid in enumerate(sids['sample_id'].values):
    if j%50 == 0: print(j, sid)
    df = conn.sql("SELECT xz_ip, xz_it FROM output.matches WHERE sample_id=%i GROUP BY xz_ip, xz_it" % sid).df()
    for i, row in df.iterrows():
        ret = conn.sql("SELECT * FROM %s.arr WHERE ip=%i AND it=%i AND id=%i" % (dbname, row.xz_ip, row.xz_it, sid)).df()
        if i == 0 and j == 0:
            conn.sql("CREATE OR REPLACE TABLE output.sample AS SELECT * FROM ret")
        else:
            conn.sql("INSERT INTO output.sample SELECT * FROM ret")

conn.close()

