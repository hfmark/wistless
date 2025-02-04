import duckdb
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
import os, sys
from utils import *

from streamlit.connections import ExperimentalBaseConnection
from streamlit.runtime.caching import cache_data

st.set_page_config(page_title="wistless", page_icon=":duck:")

class DuckDBConnection(ExperimentalBaseConnection[duckdb.DuckDBPyConnection]):
    """ DuckDB experimental connection using cloud access for db file
    """

    def _connect(self, **kwargs) -> duckdb.DuckDBPyConnection:
        if 'database' in kwargs:
            db = kwargs.pop('database')
        else:
            print('no db path given')
            sys.exit()
        conn = duckdb.connect(database=db, **kwargs)
        return conn
    
    def cursor(self) -> duckdb.DuckDBPyConnection:
        return self._instance.cursor()

    def query(self, query: str, ttl: int = 3600, **kwargs) -> pd.DataFrame:
        @cache_data(ttl=ttl)
        def _query(query: str, **kwargs) -> pd.DataFrame:
            cursor = self.cursor()
            cursor.execute(query, **kwargs)
            return cursor.df()
        
        return _query(query, **kwargs)

    def sql(self,query: str):
        return self._instance.sql(query)

def get_db_connection(database='Data/hacker_all.db',read=True):
    """ Wrapper function to connect to database
    """
    if "duck_conn" not in st.session_state:
        st.session_state['duck_conn'] = DuckDBConnection(database=database, connection_name='duck', read_only=read)
    return st.session_state["duck_conn"]

def main():
    """Initialize db connection and page when this script is run
    """
    conn = get_db_connection()
    create_side_bar(conn)
    create_page(conn)


def create_side_bar(conn: duckdb.DuckDBPyConnection):
    """ Sidebar design

    currently doc text, markdown
    """
    with st.sidebar:

        st.markdown("# how to use this tool")
        st.write("Query the database for samples that meet specified conditions using the query builder. After running a query, you can visualize aspects of the results and do some very minimal property fitting in the other tabs. For more information, see the 'documentation' tab")

def create_page(conn: duckdb.DuckDBPyConnection):
    """ Page design
    """
    st.title("wistless :duck:")
    st.write("(Whole-rock Interpretive Seismic Toolbox for LowEr cruStal Samples)")
    st.divider()

    cur = conn.cursor()

    skip_these = ['ip','id','it','meh']  # columns that should not be returnable or filterable
    # (P and T are dealt with separately, don't need ip or it directly, and id is for internal use)
    # TODO mdb_name probably not meaningful for most people, could return but not filter?
    arr_vars = [e[0] for e in cur.execute("describe hacker_all.arr").fetchall() if e[0] not in skip_these]
    dtypes = dict(conn.sql("select column_name, data_type from information_schema.columns").fetchall())

    minP,maxP,minT,maxT = conn.sql("SELECT min(pressure), max(pressure), min(temperature), max(temperature) FROM hacker_all.pt").fetchall()[0]

    tab_build, tab_plot, tab_calc, tab_doc = st.tabs(['query builder','data viz','(mis)fitting','documentation'])

    # set up some text in the session state to write and update in spots
    if 'tx_bestT' not in st.session_state:
        st.session_state['tx_bestT'] = "best fit T:"
    if 'tx_fitted' not in st.session_state:
        st.session_state['tx_fitted'] = "fitted mean [], stdev [], #points []"
    if 'tx_nret' not in st.session_state:
        st.session_state['tx_nret'] = "points returned: "

########################################################################
    # query builder
    with tab_build:

        with st.form('filterform'):
            to_filter = st.multiselect('fields to filter on',arr_vars)
            nf = len(to_filter)
            st.form_submit_button('set filters')

        inputs = [st.columns(3) for i in range(nf)]
        rads = []
        vals = []
        pfilt = False; tfilt = False
        for ic,ccc in enumerate(inputs):
            ccc[0].write(to_filter[ic])  # the thing we are filtering on
            if dtypes[to_filter[ic]] != "VARCHAR":  # numeric data types

                if to_filter[ic] in ['pressure','temperature']:
                    rad_options = ['+-','%','<',">",'<=','>=','=']
                    
                    if to_filter[ic] == 'pressure': pfilt = True
                    if to_filter[ic] == 'temperature': tfilt = True
                else:
                    rad_options = ['+-','%','<',">",'<=','>=','=','!=','in']

                rad = ccc[1].selectbox('condition type',key='radio_%i' % ic,options=rad_options)
            
                # get the numeric range so we don't set anything weird
                minV,maxV = conn.sql("SELECT min(%s), max(%s) FROM hacker_all.arr WHERE %s >= 0" % \
                                    (to_filter[ic],to_filter[ic],to_filter[ic])).fetchall()[0]
                if rad == 'in':  # range slider
                    val = ccc[2].slider('range',min_value=minV,max_value=maxV,value=(minV,maxV))
                elif rad in ['+-','%']:
                    v0 = ccc[2].number_input('center',key='val0_%i' % ic,min_value=minV,max_value=maxV)
                    v1 = ccc[2].number_input('range',key='val1_%i' % ic,\
                                            min_value=0.,max_value=100.,step=0.01)
                    val = (v0,v1)
                else:  # simple conditional
                    val = ccc[2].number_input('value',key='value_%i' % ic,min_value=minV,max_value=maxV)

            elif dtypes[to_filter[ic]] == "VARCHAR":  # string data types
                rad = ccc[1].radio('condition type',key='radio_%i' % ic,options=['=','!=']) 
                val = ccc[2].text_input('string',key='value_%i' % ic)

            rads.append(rad)
            vals.append(val)

        st.session_state['filts'] = to_filter
        st.session_state['rads'] = rads
        st.session_state['vals'] = vals  # save for joint misfit calc

        # returns, P/T sliders
        if not pfilt:
            (p_lo,p_hi) = st.slider('pressure range, GPa',min_value=minP,max_value=maxP,\
                                    value=(minP,maxP),key='pressure_slider')
        else:
            p_lo, p_hi = minP, maxP
        if not tfilt:
            (t_lo,t_hi) = st.slider('temperature range, C',min_value=minT,max_value=maxT,\
                                    value=(minT,maxT),key='temper_slider')
        else:
            t_lo, t_hi = minT, maxT
        to_return  = st.multiselect('fields to return',arr_vars)
                
        # build the query from all of the things
        ands = []
        # look at the list of fields to return
        arr_ret = [e for e in to_return]

        # handle PT conditions: at limits? single value? a set range?
        if p_lo != minP or p_hi != maxP:
            to_filter.append('pressure')
            rads.append('in')
            vals.append((p_lo, p_hi))
        if t_lo != minT or t_hi != maxT:
            to_filter.append('temperature')
            rads.append('in')
            vals.append((t_lo, t_hi))

        # make ands from the filter lists
        arr_ret.append('id')
        q1 = construct_query_new(cur, to_filter, rads, vals, arr_ret, dtypes, [minP,maxP,minT,maxT])

        with st.expander("view the query"):
            st.write(q1)

        if st.button("run query"):
            try:
                df = runquery(cur,q1)
                st.session_state['pt_df'] = df  # save for later! can in theory then plot?
                st.toast("%i query results saved to session" % len(df),icon="🦆")
                st.session_state["tx_nret"] = "points returned: %i" % len(df)
            except Exception as e:
                st.error(e)
        st.write(st.session_state["tx_nret"])

        if 'pt_df' in st.session_state.keys():  # button only visible when download is possible
            st.download_button("download results",data=convert_df(st.session_state['pt_df']),\
                                file_name='pt_results.csv',mime='text/csv')

    with tab_plot:
        if 'pt_df' in st.session_state.keys():
            avail_cols = st.session_state['pt_df'].columns
            avail_cols_num = [c for c in st.session_state['pt_df'].columns if st.session_state['pt_df'][c].dtype in [float,int,'float32','float64','int64','int32']]
            avail_cols_txt = [c for c in st.session_state['pt_df'].columns if st.session_state['pt_df'][c].dtype in ['O',str]]
        else:
            avail_cols = []
            avail_cols_num = []
            avail_cols_txt = []
        p_scatter,p_hist,p_heatmap,p_textpie = st.tabs(['scatter plot','histogram','P/T heatmap','category pie chart'])
        with p_scatter:
            # scatter plot, for now
            xax = st.selectbox("x axis quantity",avail_cols)
            yax = st.selectbox("y axis quantity",avail_cols)
            if st.button("make scatter plot"):
                st.scatter_chart(data=st.session_state['pt_df'],x=xax,y=yax)

        with p_hist:
            hxax = st.selectbox("histogram quantity",avail_cols_num)  # no text hist, they don't work
            nbins = st.select_slider("max number of bins?",options=np.arange(10,31))
            if st.button("make histogram"):
                st.altair_chart(alt.Chart(st.session_state['pt_df']).mark_bar().encode(alt.X(hxax).bin(maxbins=nbins),y='count()',))

        with p_heatmap:
            if 'pressure' in avail_cols and 'temperature' in avail_cols:
                if st.button("plot P/T heatmap"):
                    st.altair_chart(alt.Chart(st.session_state['pt_df']).mark_rect().encode(alt.X('pressure:Q').bin(),alt.Y('temperature:Q').bin(),alt.Color('count():Q').scale(scheme='greenblue')))
            else:
                st.write('need pressure and temperature returned to make this heatmap')

        with p_textpie:
            if len(avail_cols_txt) > 0:
                pie = st.selectbox("thing to chart",avail_cols_txt)
                if st.button("make chart"):
                    st.altair_chart(alt.Chart(st.session_state['pt_df']).mark_arc(innerRadius=50).encode(theta=alt.Theta("count():Q"),color=alt.Color(pie,type='nominal')))

    with tab_calc:
        # joint misfit: vp, vs, vpvs only (and = or in a range)
        mis_calc = 'To calculate misfit, filter the database on at least one of: vp, vs, vpvs'
        if 'filts' in st.session_state.keys() and 'pt_df' in st.session_state.keys() and \
                len(st.session_state['pt_df']) > 0:
            if 'vp' in st.session_state['filts'] or 'vs' in st.session_state['filts'] or \
                    'vpvs' in st.session_state['filts']:
                st.session_state['pt_df'] = joint_misfit(st.session_state['pt_df'],\
                                            st.session_state['filts'],st.session_state['vals'],\
                                            st.session_state['rads'])
                mis_calc = 'misfits calculated: '
                mis_calc += ', '.join([e.lstrip('misfit_') for e in st.session_state['pt_df'].columns\
                                         if e.startswith('misfit')])
                mis_calc += ', joint' # there will be one even if it's just one column and itself
        st.write(mis_calc)

        # best fit T: gaussian or min misfit
        bfT_rad = st.radio('Type of T fitting: ', options=['gaussian','joint misfit'])
        if st.button('calculate best fit T'):
            if 'pt_df' in st.session_state.keys() and 'temperature' in st.session_state['pt_df'].columns:
                if bfT_rad == 'gaussian':
                        best_T = gaussian_best_T(st.session_state['pt_df'])
                        st.session_state["tx_bestT"] = "best fit T: %.2f" % best_T
                else:
                    if 'joint_misfit' in st.session_state['pt_df'].columns:
                        best_T = st.session_state['pt_df'].iloc[st.session_state['pt_df']['joint_misfit'].argmin()]['temperature']
                        st.session_state["tx_bestT"] = "best fit T: %.2f" % best_T
                    else:
                        st.toast('joint misfit required for this calculation')
            else:
                st.toast('run a query that returns temperature, and calculate misfit, before fitting T')
        st.write(st.session_state["tx_bestT"])

        # misfit-weighted mean and stdev for some property
        avail_cols_num = []
        if 'pt_df' in st.session_state.keys() and 'pressure' in st.session_state['pt_df'].columns and 'temperature' in st.session_state['pt_df'].columns and len(st.session_state['pt_df']) > 0:
            avail_cols_num = [c for c in st.session_state['pt_df'].columns if st.session_state['pt_df'][c].dtype in [float,int,'float32','float64','int64','int32'] and not c.startswith('misfit') and not c.endswith('misfit') and c not in ['id','temperature','pressure']]
            c1,c2,c3 = st.columns(3)
            pr_unique = np.unique(st.session_state['pt_df']['pressure'],return_counts=True)
            tm_unique = np.unique(st.session_state['pt_df']['temperature'],return_counts=True)
            with c1:
                propfit = st.selectbox('property to fit',avail_cols_num)
            with c2:
                pr_ops = ['%.2f (%i)' % (pr_unique[0][i], pr_unique[1][i]) for i in range(len(pr_unique[0]))]
                pfit = st.select_slider('fit P: value (#points)',options = pr_ops)
            with c3:
                tm_ops = ['%.2f (%i)' % (tm_unique[0][i], tm_unique[1][i]) for i in range(len(tm_unique[0]))]
                tfit = st.select_slider('fit T: value (#points)',options = tm_ops)
            if st.button("fit property"):
                # recover the closest pressure/temperature from the selectors
                pfit_actual = pr_unique[0][np.argmin(abs(pr_unique[0] - float(pfit.split('(')[0])))]
                tfit_actual = tm_unique[0][np.argmin(abs(tm_unique[0] - float(tfit.split('(')[0])))]
                sel = st.session_state['pt_df'][(st.session_state['pt_df']['pressure'] == pfit_actual)&(st.session_state['pt_df']['temperature'] == tfit_actual)]
                if len(sel) > 1 and 'joint_misfit' in sel.columns:
                    mean_val = np.average(sel[propfit],weights=1./sel['joint_misfit'])
                    std_val = np.sqrt(np.cov(sel[propfit], aweights=1./sel['joint_misfit']))
                    st.session_state['tx_fitted'] = 'mean: %.3f, std: %.3f, #points: %i' % (mean_val,std_val,len(sel))
                else:
                    st.toast('not enough samples in range to fit')
                st.write(st.session_state['tx_fitted'])

    with tab_doc:
        lotsofdocs = read_markdown_file('wistless-docs.md')
        st.markdown(lotsofdocs, unsafe_allow_html=True)





if __name__ == "__main__":
    main()


