import duckdb
import streamlit as st
import numpy as np
#from code_editor import code_editor
import altair as alt
import os, sys

st.set_page_config(page_title="wistless", page_icon=":duck:")

# @st.cache_resource
def get_db_connection():
    """ connect to database, hard-coded file and read-only
    """
    if "duck_conn" not in st.session_state:
        st.session_state["duck_conn"] = duckdb.connect(database="Data/hacker_noamph_denorm.db",read_only=True)

    return st.session_state["duck_conn"]

def main():
    """ initialize db connection and page when this script is run
    """
    conn = get_db_connection()
    create_side_bar(conn)
    create_page(conn)


def create_side_bar(conn: duckdb.DuckDBPyConnection):
    """ sidebar design

    currently a list of tables and their columns/data types
    """
    cur = conn.cursor()

    with st.sidebar:
        #st.divider()

        st.markdown("# how to use this tool")
        st.write('Query the database for samples that meet specified conditions using the query builder. After running a query, you can export the results to a file and/or visualize aspects of the results. T and property fitting will be added at some point.')

        #table_list = ""
        #cur.execute("show all tables")
        #recs = cur.fetchall()

        #if len(recs) > 0:
        #    st.markdown("# tables")

        #for rec in recs:
        #    table_name = rec[2]
        #    if table_name != 'sqlite_sequence':
        #        table_list += f"- {table_name}\n"
        #        cur.execute(f"describe {table_name}")

        #        for col in cur.fetchall():
        #            table_list += f"    - {col[0]} {col[1]}\n"

        #st.markdown(table_list)

def create_page(conn: duckdb.DuckDBPyConnection):
    """ page design
    """
    st.title("wistless :duck:")
    st.write("(Whole-rock Interpretive Seismic Toolbox for LowEr cruStal Samples)")
    st.divider()

    cur = conn.cursor()

    skip_these = ['ip','id','it','meh']  # columns that should not be returnable or filterable
    # (P and T are dealt with separately, don't need ip or it directly, and id is for internal use)
    arr_vars = [e[0] for e in cur.execute("describe arr").fetchall() if e[0] not in skip_these]
    dtypes = dict(conn.sql("select column_name, data_type from information_schema.columns").fetchall())

    minP,maxP,minT,maxT = conn.sql("SELECT min(pres), max(pres), min(temp), max(temp) FROM pt").fetchall()[0]

    #tab_build, tab_write, tab_plot = st.tabs(['query builder','direct SQL','data viz'])
    tab_build, tab_plot = st.tabs(['query builder','data viz'])

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
                rad = ccc[1].radio('condition type',key='radio_%i' % ic,options=['<',r"\>",'<=',r'\>=','=','!=','in'])
                # get the numeric range so we don't set anything weird
                minV,maxV = conn.sql("SELECT min(%s), max(%s) FROM arr WHERE %s >= 0" % (to_filter[ic],to_filter[ic],to_filter[ic])).fetchall()[0]
                if rad == 'in':  # range slider
                    val = ccc[2].slider('range',min_value=minV,max_value=maxV,value=(minV,maxV))
                else:  # simple conditional
                    val = ccc[2].number_input('value',key='value_%i' % ic,min_value=minV,max_value=maxV)
            if dtypes[to_filter[ic]] == "VARCHAR":  # string data types
                rad = ccc[1].radio('condition type',key='radio_%i' % ic,options=['=','!=']) 
                val = ccc[2].text_input('string',key='value_%i' % ic)

            rads.append(rad)
            vals.append(val)

        # returns, P/T sliders
        (p_lo,p_hi) = st.slider('pressure range, GPa',min_value=minP,max_value=maxP,value=(minP,maxP))
        (t_lo,t_hi) = st.slider('temperature range, C',min_value=minT,max_value=maxT,value=(minT,maxT))
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
            if rads[i] in ['<',r"\>",'<=',r'\>=','=','!='] and dtypes[to_filter[i]] != "VARCHAR":
                ands.append("%s %s %.2f" % (to_filter[i],rads[i].lstrip('\\'),vals[i]))
                if rads[i] in ['<','<=']:
                    ands.append("%s %s 0" % (to_filter[i],r'>='))  # screen out -999s that are nulls

            elif rads[i] == 'in':  # this is not an option for strings, and it's slider only right now
                ands.append("%s between %f and %f" % (to_filter[i],vals[i][0],vals[i][1]))
            elif rads[i] in  ['=','!='] and dtypes[to_filter[i]] == "VARCHAR":
                ands.append("%s %s '%s'" % (to_filter[i],rads[i],vals[i]))
        arr_ret.append('id')
        q1 = "SELECT %s FROM arr WHERE " % (', '.join(arr_ret))
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

        #@st.fragment()
        #def download_dfs():
        if st.button("download results",key='qb_download'):
            if 'pt_df' in st.session_state.keys():
                st.session_state['pt_df'].to_csv("pt_results.csv",index=False)
                st.toast("query results written to file",icon="🦆")
                st.balloons()
        #download_dfs()

#########################################################################
#    # DIY SQL
#    with tab_write:
#        st.write("enter your query below")
#        #st.write("ctrl+enter to run the SQL")
#        custom_buttons = [ {"name": "Submit",
#               "feather": "Play",
#               "alwaysOn": True,
#               "primary": True,
#               "hasText": True,
#               "showWithIcon": True,
#               "commands": ["submit"],
#               "style": {"bottom": "0.44rem", "right": "0.4rem"}
#             },]
#        res = code_editor(code="", lang="sql", key="editor", buttons=custom_buttons, height='100px')
#
#        # tabular printout, download
#        st.write('query results:')
#        for query in res["text"].split(";"):
#            if query.strip() == "":
#                continue
#            try:
#                cur.execute(query)
#                df = cur.fetch_df()
#                st.session_state['pt_df'] = df  # save for later! can in theory then plot?
#                st.toast("%i query results saved to session" % len(df),icon="🦆")
#            except Exception as e:
#                st.error(e)
#
#        @st.fragment()
#        def download_dfs():
#            if st.button("download results",key='sql_download'):
#                if 'pt_df' in st.session_state.keys():
#                    st.session_state['pt_df'].to_csv("pt_results.csv",index=False)
#                    st.balloons()
#        download_dfs()


    with tab_plot:
        #if 'pt_df' not in st.session_state.keys():
        #    st.write("no query results to plot - go run a query first")
        #else:
        if st.button("check what's available"):
            if 'pt_df' not in st.session_state.keys():
                st.write("no query results to plot - go run a query first")
            else:
                avail_cols = st.session_state['pt_df'].columns
                st.write("%i query results" % len(st.session_state['pt_df']))
                st.write("available columns are %s" % (', '.join(avail_cols)))

        if 'pt_df' in st.session_state.keys():
            avail_cols = st.session_state['pt_df'].columns
        else:
            avail_cols = []
        p_scatter,p_hist,p_heatmap = st.tabs(['scatter plot','histogram','P/T heatmap'])
        with p_scatter:
            # scatter plot, for now
            xax = st.selectbox("x axis quantity",avail_cols)
            yax = st.selectbox("y axis quantity",avail_cols)

            @st.fragment()
            def scatter_plot():
                if st.button("make scatter plot"):
                    st.scatter_chart(data=st.session_state['pt_df'],x=xax,y=yax)
            scatter_plot()

        with p_hist:
            hxax = st.selectbox("histogram quantity",avail_cols)

            @st.fragment()
            def hist_plot():
                if st.button("make histogram"):
                    st.altair_chart(alt.Chart(st.session_state['pt_df']).mark_bar().encode(alt.X(hxax,bin=True),y='count()',))
            hist_plot()

        with p_heatmap:
            if 'pres' in avail_cols and 'temp' in avail_cols:
                @st.fragment()
                def ptheat_plot():
                    if st.button("plot P/T heatmap"):
                        st.altair_chart(alt.Chart(st.session_state['pt_df']).mark_rect().encode(alt.X('pres:Q').bin(),alt.Y('temp:Q').bin(),alt.Color('count():Q').scale(scheme='greenblue')))
                ptheat_plot()
            else:
                st.write('need pres and temp returned to make this heatmap')

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
        qp = "SELECT id FROM pt WHERE %s BETWEEN ? AND ? ORDER BY abs(?-%s) LIMIT 1" % (col[pt],col[pt])
        ipt = cursor.execute(qp,(low,hgh,tofit[0])).fetchall()[0][0]
        ands = "%s = %i " % (ii[pt],ipt)
    elif len(tofit) == 2:
        ands = "%s IN (SELECT id FROM pt WHERE %s BETWEEN %f AND %f)" % (ii[pt],col[pt],tofit[0],tofit[1])
        qp = "SELECT id FROM pt WHERE %s BETWEEN ? AND ?" % (col[pt])
        _ = cursor.execute(qp,(tofit[0],tofit[1]))
        ipt = np.array([e[0] for e in cursor.fetchall()])
    if return_and:
        return ands
    else:
        return ipt



if __name__ == "__main__":
    main()

