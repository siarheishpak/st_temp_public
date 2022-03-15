import streamlit as st
import pandas as pd
import pyarrow as pa
import numpy as np

if st.checkbox('pyarrow with schema (based on pd.DataFrame)'):
    with st.echo():
        df = pd.DataFrame({
                'int': [1, 1, 2, 3, 5],
                'str': ['a', 'b', 'c', 'ab', 'bc']
        })

    st.write('No schema')
    with st.echo():
        table1 = pa.Table.from_pandas(df, columns=['str'])
        st.dataframe(table1)

    st.write('Schema is manually provided')
    with st.echo():
        fields = [pa.field('int', pa.int64()), pa.field('str', pa.string())]
        schema = pa.schema(fields)
        table2 = pa.Table.from_pandas(df, schema=schema)
        st.dataframe(table2)

    st.write('Schema is generated based on DataFrame')
    with st.echo():
        schema_from_pd = pa.Schema.from_pandas(df)
        table3 = pa.Table.from_pandas(df, schema=schema_from_pd)
        st.dataframe(table3)


if st.checkbox('pyarrow with schema (based on dict, array, pylist)'):
    st.write('Constructing pyarrow.Table from dict')
    with st.echo():
        pyarrow_dict = {'int': [1, 2], 'str': ['a', 'b']}
        pyarrow_schema3 = pa.schema([('int', pa.int64()), ('str', pa.string())])
        table3 = pa.Table.from_pydict(pyarrow_dict, schema=pyarrow_schema3)
        st.dataframe(table3)


    st.write('Constructing pyarrow.Table from arrays')
    with st.echo():
        pyarrow_array2 = [pa.array([1, 2, 3]), pa.array([4, 5, 6]), pa.array([7, 8, 9])]
        pyarrow_schema2 = pa.schema([('a', pa.int64()), ('b', pa.int64()), ('c', pa.int64())])
        table2 = pa.Table.from_arrays(pyarrow_array2, schema=pyarrow_schema2) # names=['a', 'b', 'c']
        st.dataframe(table2)

    st.write('Constructing pyarrow.Table from pylist')
    with st.echo():
        pylist = [{'int': 1, 'str': 'a'}, {'int': 2, 'str': 'b'}]
        fields4 = [pa.field('int', pa.int64()), pa.field('str', pa.string())]
        schema4 = pa.schema(fields4)
        table4 = pa.Table.from_pylist(pylist, schema=schema4)
        st.dataframe(table4)


if st.checkbox('pd.DataFrame: timedelta64'):
    with st.echo():
        st.dataframe(pd.DataFrame(
            {"timedelta64":[np.timedelta64(i, 'h') for i in range(10)]}
        ))


if st.checkbox('pd.DataFrame: float128'):
    with st.echo():
        st.dataframe(pd.DataFrame(
            {"float128_col": np.array(np.random.rand(5), dtype='g')} # dtype='g' means float128
        ))

if st.checkbox('pd.DataFrame: complex64,128,256'):
    with st.echo():
        st.dataframe(pd.DataFrame(
            {"complex": np.array([1+2j, 3+4j, 5+6j], dtype='cdouble')} # same for dtype='csingle' or 'cdouble' or 'clongdouble'
        ))
