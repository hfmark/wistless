import duckdb
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import os, sys

from streamlit.connections import ExperimentalBaseConnection
from streamlit.runtime.caching import cache_data

st.set_page_config(page_title="wistless", page_icon=":duck:")

class DuckDBConnection(ExperimentalBaseConnection[duckdb.DuckDBPyConnection]):
    """Basic st.experimental_connection implementation for DuckDB"""

    def _connect(self, **kwargs) -> duckdb.DuckDBPyConnection:
        motherduck_token = os.getenv("motherduck_token")
        conn = duckdb.connect(f"""md:hacker_noamph?motherduck_token={motherduck_token}""")
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

def get_db_connection():
    """ connect to database, hard-coded file and read-only
    """
    if "duck_conn" not in st.session_state:
        #st.session_state["duck_conn"] = duckdb.connect(database="Data/hacker_noamph_denorm.db",read_only=True)
        #st.session_state['duck_conn'] = DuckDBConnection(connection_name='duck',database='Data/hacker_noamph_denorm.db',read_only=True)
        st.session_state['duck_conn'] = DuckDBConnection(connection_name='duck',read_only=True)
    return st.session_state["duck_conn"]

def main():
    """ initialize db connection and page when this script is run
    """
    conn = get_db_connection()
    create_side_bar(conn)
    create_page(conn)


def create_side_bar(conn: duckdb.DuckDBPyConnection):
    """ sidebar design
    currently doc text
    """
    with st.sidebar:

        st.markdown("# how to use this tool")
        st.write('Query the database for samples that meet specified conditions using the query builder. After running a query, you can export the results to a file and/or visualize aspects of the results. T and property fitting will be added at some point.')
        st.divider()
        st.markdown("## the database")
        st.write("Lower crustal samples run through perpleX. Each sample has associated text quantities (name, rock type, per/metaluminous flag), bulk composition (and associated wt%SiO2, Mg#), and calculated quantities (vp, vs, density on a grid of P and T).")
        st.markdown("## filter conditions")
        st.write("Pretty much all values can be used to filter the database. The most useful filter types are usually +-, where the user sets a center value and +/- range; and %, where the user sets a center value and percentage range. Other standard conditionals (<, >, =) are available. The 'in' option lets users select a specific range for a quantity within the full range of values in the database. For bulk composition this may be helpful; for calculated things like vp and vs, there are some extreme outlier values that make the sliders less useful.")
        st.markdown("## returns")
        st.write("Users can specify which quantities to return. Those are the things that will be output to a file and that are made available for plotting in the data viz tab.")

def create_page(conn: duckdb.DuckDBPyConnection):
    """ page design
    """
    st.title("wistless :duck:")
    st.write("(Whole-rock Interpretive Seismic Toolbox for LowEr cruStal Samples)")
    st.divider()

    cur = conn.cursor()

    skip_these = ['ip','id','it','meh']  # columns that should not be returnable or filterable
    # (P and T are dealt with separately, don't need ip or it directly, and id is for internal use)
    arr_vars = [e[0] for e in cur.execute("describe hacker_noamph.arr").fetchall() if e[0] not in skip_these]
    dtypes = dict(conn.sql("select column_name, data_type from information_schema.columns").fetchall())

    minP,maxP,minT,maxT = conn.sql("SELECT min(pres), max(pres), min(temp), max(temp) FROM hacker_noamph.pt").fetchall()[0]

    tab_build, tab_plot, tab_calc = st.tabs(['query builder','data viz','(mis)fitting'])

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
        for ic,ccc in enumerate(inputs):
            ccc[0].write(to_filter[ic])  # the thing we are filtering on
            if dtypes[to_filter[ic]] != "VARCHAR":  # numeric data types
                rad = ccc[1].selectbox('condition type',key='radio_%i' % ic,options=['+-','%','<',">",'<=','>=','=','!=','in'])
                # get the numeric range so we don't set anything weird
                minV,maxV = conn.sql("SELECT min(%s), max(%s) FROM hacker_noamph.arr WHERE %s >= 0" % (to_filter[ic],to_filter[ic],to_filter[ic])).fetchall()[0]
                if rad == 'in':  # range slider
                    val = ccc[2].slider('range',min_value=minV,max_value=maxV,value=(minV,maxV))
                elif rad in ['+-','%']:
                    v0 = ccc[2].number_input('center',key='val0_%i' % ic,min_value=minV,max_value=maxV)
                    v1 = ccc[2].number_input('range',key='val1_%i' % ic,min_value=0.,max_value=100.,step=0.01)
                    val = (v0,v1)
                else:  # simple conditional
                    val = ccc[2].number_input('value',key='value_%i' % ic,min_value=minV,max_value=maxV)
            if dtypes[to_filter[ic]] == "VARCHAR":  # string data types
                rad = ccc[1].radio('condition type',key='radio_%i' % ic,options=['=','!=']) 
                val = ccc[2].text_input('string',key='value_%i' % ic)

            rads.append(rad)
            vals.append(val)

        st.session_state['filts'] = to_filter
        st.session_state['rads'] = rads
        st.session_state['vals'] = vals  # save for joint misfit calc

        # returns, P/T sliders
        (p_lo,p_hi) = st.slider('pressure range, GPa',min_value=minP,max_value=maxP,value=(minP,maxP),key='pressure_slider')
        (t_lo,t_hi) = st.slider('temperature range, C',min_value=minT,max_value=maxT,value=(minT,maxT),key='temper_slider')
        to_return  = st.multiselect('fields to return',arr_vars)
                
        # build the query from all of the things
        ands = []
        # look at the list of fields to return
        arr_ret = [e for e in to_return]

        # handle PT conditions: at limits? single value? a set range?
        if p_lo != minP or p_hi != maxP:
            ands.append(pt_select(cur,'p',[p_lo,p_hi],return_and=True))
        if t_lo != minT or t_hi != maxT:
            ands.append(pt_select(cur,'t',[t_lo,t_hi],return_and=True))

        # make ands from the filter lists
        for i in range(len(to_filter)):
            if rads[i] in ['<',">",'<=','>=','=','!='] and dtypes[to_filter[i]] != "VARCHAR":
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
            elif rads[i] == 'in':  # this is not an option for strings, and it's slider only right now
                ands.append("%s between %f and %f" % (to_filter[i],vals[i][0],vals[i][1]))
            elif rads[i] in  ['=','!='] and dtypes[to_filter[i]] == "VARCHAR":
                ands.append("%s %s '%s'" % (to_filter[i],rads[i],vals[i]))
        arr_ret.append('id')
        q1 = "SELECT %s FROM hacker_noamph.arr WHERE " % (', '.join(arr_ret))
        for i,a in enumerate(ands):
            q1 += a
            if i != len(ands)-1:
                q1 += " AND "

        with st.expander("view the query"):
            st.write(q1)

        if st.button("run query"):
            try:
                cur.execute(q1)
                df = cur.fetch_df()
                st.session_state['pt_df'] = df  # save for later! can in theory then plot?
                st.toast("%i query results saved to session" % len(df),icon="🦆")
            except Exception as e:
                st.error(e)

        if st.button("download results",key='qb_download'):
            if 'pt_df' in st.session_state.keys():
                st.session_state['pt_df'].to_csv("pt_results.csv",index=False)
                st.toast("query results written to file",icon="🦆")
                st.balloons()


        if st.button('reset everything'):
            conn._instance.close()
            for k in st.session_state.keys():
                del st.session_state[k]
            st.rerun()



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
            if st.button("make histogram"):
                st.altair_chart(alt.Chart(st.session_state['pt_df']).mark_bar().encode(alt.X(hxax,bin=True),y='count()',))

        with p_heatmap:
            if 'pres' in avail_cols and 'temp' in avail_cols:
                if st.button("plot P/T heatmap"):
                    st.altair_chart(alt.Chart(st.session_state['pt_df']).mark_rect().encode(alt.X('pres:Q').bin(),alt.Y('temp:Q').bin(),alt.Color('count():Q').scale(scheme='greenblue')))
            else:
                st.write('need pres and temp returned to make this heatmap')

        with p_textpie:
            if len(avail_cols_txt) > 0:
                pie = st.selectbox("thing to chart",avail_cols_txt)
                if st.button("make chart"):
                    st.altair_chart(alt.Chart(st.session_state['pt_df']).mark_arc(innerRadius=50).encode(theta=alt.Theta("count():Q"),color=alt.Color(pie,type='nominal')))

    with tab_calc:
        # joint misfit: vp, vs, vpvs only (and = or in a range)
        if st.button("calculate (joint) misfit)"):
            mis_calc = 'calculated '
            # check if vp, vs, or vpvs is in the filters with the right kind of condition
            if 'filts' in st.session_state.keys() and 'pt_df' in st.session_state.keys():
                for i in range(len(st.session_state['rads'])):  # individual misfits
                    ff = st.session_state['filts'][i]
                    vv = st.session_state['vals'][i]
                    rr = st.session_state['rads'][i]
                    if ff in ['vp','vs','vpvs'] and rr in ['in','=','+-','%'] and ff in st.session_state['pt_df'].columns:
                        if rr == 'in':
                            fitval = vv[0] + (vv[1] - vv[0])/2
                        else:
                            fitval = vv[0]
                        st.session_state['pt_df']['misfit_%s' % ff] = (st.session_state['pt_df'][ff] - fitval)/fitval
                        mis_calc += 'misfit_%s, ' % ff

                new_joint = np.zeros(len(st.session_state['pt_df']))
                for col in st.session_state['pt_df'].columns:
                    if col.startswith('misfit_'):
                        new_joint += st.session_state['pt_df'][col]**2
                st.session_state['pt_df'].loc[:,'joint_misfit'] = np.sqrt(new_joint)
                mis_calc += 'joint misfit'
                st.write(mis_calc)
            else:
                st.write('no misfit calculated; filter on vp, vs, and/or vpvs first')

        # best fit T: gaussian or min misfit  TODO min misfit, callback and columns
        if st.button('calculate best fit T'):
            if 'pt_df' in st.session_state.keys() and 'temp' in st.session_state['pt_df'].columns:
                Ts, counts = np.unique(st.session_state['pt_df']['temp'],return_counts=True)
                sum_fits = sum(counts)
                sum_M2 = sum(counts*Ts**2)
                best_T = sum(Ts*counts)/sum_fits
                st.write(best_T)
            else:
                st.write('run a query that returns temp, and calculate misfit, before fitting T')

        # misfit-weighted mean and stdev for some property
        avail_cols_num = []
        if 'pt_df' in st.session_state.keys() and 'pres' in st.session_state['pt_df'].columns and 'temp' in st.session_state['pt_df'].columns:
            avail_cols_num = [c for c in st.session_state['pt_df'].columns if st.session_state['pt_df'][c].dtype in [float,int,'float32','float64','int64','int32']]
            c1,c2,c3 = st.columns(3)
            with c1:
                propfit = st.selectbox('property to fit',avail_cols_num)
            with c2:
                pfit = st.select_slider('fit pressure',options = np.unique(st.session_state['pt_df']['pres']),format_func=lambda x:'%.2f' % x)
            with c3:
                tfit = st.select_slider('fit temperature',options = np.unique(st.session_state['pt_df']['temp']),format_func=lambda x:'%.2f' % x)
            if st.button("fit property"):
                sel = st.session_state['pt_df'][(st.session_state['pt_df']['pres'] == pfit)&(st.session_state['pt_df']['temp'] == tfit)]
                if len(sel) > 1 and 'joint_misfit' in sel.columns:
                    mean_val = np.average(sel[propfit],weights=1./sel['joint_misfit'])
                    std_val = np.sqrt(np.cov(sel[propfit], aweights=1./sel['joint_misfit']))
                    st.write(mean_val,std_val,len(sel))
                else:
                    st.write('not enough samples in range to fit')


def pt_select(cursor,pt,tofit,return_and=True):
    """ get ip or it values 
    if return_and, return the condition for sql query; if not, return the ip/it values
    """
    dp = {'p': 0.1, 't': 10}
    col = {'p':'pres','t':'temp'}
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



if __name__ == "__main__":
    main()

