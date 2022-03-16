import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import make_dataclass
import pyarrow as pa
import random
from string import ascii_uppercase, ascii_lowercase, digits
import datetime
from faker import Faker

st.markdown('''### Data source''')

st.markdown('''#### pandas.DataFrame 🐼''')
st.write('Constructing DataFrame from a dictionary')
data_dict = {'column_1': [1, 2, 3, 4], 'column_2': [1, 3, 5, 7], 'column_3': [11, 13,  17, 19]}
df = pd.DataFrame(data = data_dict)
st.dataframe(df)
st.write('\+ enforce dtype')
df_dtype=pd.DataFrame(data = data_dict, dtype=np.int8)
st.dataframe(df_dtype)

st.write('Constructing DataFrame from numpy ndarray')
df_np_ndarray = pd.DataFrame(
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    columns=['col1', 'col2', 'col3']
)
st.dataframe(df_np_ndarray)

st.write('Constructing DataFrame from numpy.random.randn')
df5 = pd.DataFrame(
    np.random.randn(50, 15),
    columns=('col %d' % i for i in range(15)))
st.dataframe(df5)

st.write('Constructing DataFrame from a numpy ndarray that has labeled columns')
data_np_ndarray_label = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)],
    dtype=[("Boston", "i4"), ("Minsk", "i4"), ("Brest", "i4")])
df_data_np_ndarray_label = pd.DataFrame(data_np_ndarray_label,
    columns=['Brest', 'Boston', 'Minsk'])
st.dataframe(df_data_np_ndarray_label)

st.write('Constructing DataFrame from dataclass')
Point = make_dataclass("Point", [("x", int), ("y", int)])
df4 = pd.DataFrame([Point(0, 0), Point(0, 3), Point(2, 3)])
st.dataframe(df4)


st.markdown('''#### pyarrow.Table 🏹''')
st.write('Constructing pyarrow.Table from pd.DataFrame')
df_arr1 = pd.DataFrame({
    'int': [1, 1, 2, 3, 5],
    'str': ['a', 'b', 'c', 'ab', 'bc'],
    'float': [3.14, 2.71, 9.98, 6.02, 1060.02]
})
table1 = pa.Table.from_pandas(df_arr1)
st.dataframe(table1)

df_arr2 = pd.DataFrame(
    np.random.randn(5, 3),
    columns=['col1', 'col2', 'col3']
)
table1_2 = pa.Table.from_pandas(df_arr2)
st.dataframe(table1_2)


st.markdown('''#### numpy.ndarray 💠''')
st.write('numpy.ndarray(shape, dtype=int, buffer=None, offset=0, order=None)')
x = np.ndarray(shape=(5,5), buffer=np.arange(40), dtype=int, offset=8*np.int_().itemsize, order='F')
st.dataframe(x)

st.write('ndarray=np.arange(N).reshape((m,l))')
x1 = np.arange(40).reshape((8, 5))
st.dataframe(x1)

st.write('ndarray=np.array')
x2 = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
st.dataframe(x2)

st.write('ndarray = np.random.randn(N, M)')
x3 = np.random.randn(15, 20)
st.dataframe(x3)

st.markdown('''#### Dictionary 📖''')
dict = {"brand 🚗": ["Ford", "KIA", "Toyota", "Tesla"],
        "model 🚙": ["Mustang", "Optima", "Corolla", "Model 3"],
        "year 📆": [1964, 2007, 2022, 2021],
        "color 🌈": ["Black ⚫", "Red 🔴", "White ⚪", "Red 🔴"],
        "emoji 🚀🚀": ["👨🏻‍🚀", "👩🏻‍🚀", "👩🏻‍🚒🚀", "👨🏻‍🚒"]}
st.dataframe(dict)

st.markdown('''#### Iterable ⏭''')
st.write('Data is a listed string list("string")')
str = 'obladi oblada'
data_str = list(str)
st.dataframe(data_str)

st.write('Data is a sorted list')
st.dataframe(sorted('obladi oblada'))

st.write('1-d tuple')
st.dataframe(("apple", "banana", "cherry", "apple", "cherry"))

st.write('2-d tuple')
st.dataframe((("apple", "banana", "cherry", "apple", "cherry"),
            ("obladi", "oblada", "life", "goes", "on"),
            ("who", "let", "the", "dogs", "out"),
            ("Kids", "!", "Are", "you", "ready")))
st.write('iter(2-d tuple)')
st.dataframe(iter((("apple", "banana", "cherry", "apple", "cherry"),
            ("obladi", "oblada", "life", "goes", "on"),
            ("who", "let", "the", "dogs", "out"),
            ("Kids", "!", "Are", "you", "ready"))))

st.write('1-d set')
st.dataframe({"apple", "banana", "cherry", "apple", "cherry"})
st.write('iter(1-d set)')
st.dataframe(iter({"apple", "banana", "cherry", "apple", "cherry"}))

st.write('2-d list')
list_1 = [[1,2,3,4,5],[-1,-2,-3,-4,-5],[10,20,30,40,50],[6,7,8,9,10]]
st.dataframe(list_1)
st.write('iter(2-d list)')
list_2 = iter(list_1)
st.dataframe(list_2)


st.markdown('''#### None 👽''')
st.dataframe()
st.dataframe(width=200, height=300)



st.markdown('''### with Width and Height''')
df = pd.DataFrame(
    np.random.randn(50, 36),
    columns = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
)
st.write('No width and height specified')
st.dataframe(df)
st.write('width=200, height=300')
st.dataframe(df, width=200, height=300)
st.write('width=300, height=200')
st.dataframe(df, width=300, height=200)
st.write('5000, 1000')
st.dataframe(df, 5000, 1000)



st.markdown('''### pd.Dataframe with different types of data''')
n_rows = 10 # number or rows in the table
random_int = random.randint(30,50) # a random number in a range (30,50)
chars = ascii_uppercase + ascii_lowercase + digits # will use it to generate strings

######################################################################
### Sparse datatype: sparse pandas data (column sparse) not supported
######################################################################
# sparse_array = np.random.randn(n_rows)
# sparse_array[2: -2] = np.nan
# sparse_data = pd.Series(pd.arrays.SparseArray(sparse_array))

# print('------------------------')
dft = pd.DataFrame(
    {
        "float64": np.random.rand(n_rows),
        "int64": np.arange(random_int,random_int + n_rows),
        "numpy bool": [random.choice([True, False]) for _ in range(n_rows)],
        "boolean": pd.array([random.choice([True, False, None]) for _ in range(n_rows)], dtype = "boolean"),
        #"timedelta64":[np.timedelta64(i+1, 'h') for i in range(n_rows)],
        "datetime64": [(np.datetime64('2022-03-11T17:13:00') - random.randint(400000,1500000)) for _ in range(n_rows)],
        "datetime64 + TZ": [(pd.to_datetime('2022-03-11 17:41:00-05:00')) for _ in range(n_rows)],
        "string_object": [''.join(random.choice(chars) for i in range(random_int)) for j in range(n_rows)],
        "string_string": [''.join(random.choice(chars) for i in range(random_int)) for j in range(n_rows)],
        "category": pd.Series(list(''.join(random.choice(ascii_lowercase) for i in range(n_rows)))).astype("category"),
        "period[H]": [(pd.Period("2022-03-14 11:52:00", freq="H") + pd.offsets.Hour(i)) for i in range(n_rows)],
        # "sparse": sparse_data # Sparse pandas data (column sparse) not supported
        "interval": [pd.Interval(left=i, right=i+1, closed='both') for i in range(n_rows)]
    }
)
dft = dft.astype({"string_string":"string"}) # string_string initially had the 'object' dtype. this line convert it into 'string'
# print(dft.dtypes)
st.dataframe(dft)

st.write("Interval dtype in pd.DataFrame")
interval_df = pd.DataFrame(
    {
        "int64_both": [pd.Interval(left=i, right=i+1, closed='both') for i in range(n_rows)],
        "int64_right": [pd.Interval(left=i, right=i+1, closed='right') for i in range(n_rows)],
        "int64_left": [pd.Interval(left=i, right=i+1, closed='left') for i in range(n_rows)],
        "int64_neither": [pd.Interval(left=i, right=i+1, closed='neither') for i in range(n_rows)],
        "timestamp_right_defualt": [pd.Interval(left=pd.Timestamp(2022, 3, 14, i), right=pd.Timestamp(2022, 3, 14, i+1)) for i in range(n_rows)],
        "float64": [pd.Interval(random.random(), random.random() + 1) for _ in range(n_rows)]
    }
)
# print(interval_df.dtypes)
st.dataframe(interval_df)

st.write("numeric dtypes in pd.DataFrame")
int_df = pd.DataFrame(
    {
        "int64": pd.array([1, 2, 3, 4, 5], dtype="Int64"),
        "int32": pd.array([1, 2, 3, 4, 5], dtype="Int32"),
        "int16": pd.array([1, 2, 3, 4, 5], dtype="Int16"),
        "int8": pd.array([1, 2, 3, 4, 5], dtype="Int8"),
        "uint64": pd.array([1, 2, 3, 4, 5], dtype="UInt64"),
        "uint32": pd.array([1, 2, 3, 4, 5], dtype="UInt32"),
        "uint16": pd.array([1, 2, 3, 4, 5], dtype="UInt16"),
        "uint8": pd.array([1, 2, 3, 4, 5], dtype="UInt8"),
        # "float128": np.array(np.random.rand(5), dtype='g'), # dtype='g' means float128
        "float64": np.random.rand(5),
        "float32": pd.array(np.random.rand(5), dtype="float32"),
        "float16": pd.array(np.random.rand(5), dtype="float16")
    }
)
# print(int_df.dtypes)
st.dataframe(int_df)



st.markdown('''### pd.DataFrame with indexes''')
st.markdown('''#### Single index  🚹''')
st.write('Example1')
dates = pd.date_range('1/1/2000', periods=8)
df_index1 = pd.DataFrame(
    np.random.randn(8, 4),
    index=dates,
    columns=['A', 'B', 'C', 'D']
)
st.dataframe(df_index1)

st.write('Example2')
df_index2 = pd.DataFrame(pd.Series([1, 2, 3], index=list('abc')))
st.dataframe(df_index2)

st.write('Example3: + `.loc`')
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
df_index3 = pd.DataFrame(
    np.random.randn(6, 4),
    index=list('abcdef'),
    columns=list('ABCD')
)
with col1:
    st.dataframe(df_index3)
df_index3_loc1 = df_index3.loc[['a', 'b', 'd'], :]
with col2:
    st.dataframe(df_index3_loc1)
df_index3_loc2 = df_index3.loc['e':, 'B':]
with col3:
    st.dataframe(df_index3_loc2)
df_index3_loc3 = df_index3.loc['a']
with col4:
    st.dataframe(df_index3_loc3)

st.write('Example4: + `.iloc`')
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)
df_index4 = pd.DataFrame(
    np.random.randn(6, 4),
    index=list(range(0, 12, 2)),
    columns=list(range(10, 50, 10))
)
with col5:
    st.dataframe(df_index4)
df_index4_iloc1 = df_index4.iloc[:3]
with col6:
    st.dataframe(df_index4_iloc1)
df_index4_iloc2 = df_index4.iloc[1:5, 2:4]
with col7:
    st.dataframe(df_index4_iloc2)
df_index4_iloc3 = df_index4.iloc[[1, 3, 5], [1, 3]]
with col8:
    st.dataframe(df_index4_iloc3)

st.write('Example5')
col9, col10, col11 = st.columns(3)
df_index5 = pd.DataFrame(
    {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    },
    index=['abc', 'wololo', 'green tea']
)
with col9:
    st.dataframe(df_index5)
with col10:
    st.dataframe(df_index5.iloc[[0, 2], df_index5.columns.get_indexer(['A', 'C'])])
with col11:
    st.dataframe(df_index5.loc[df_index5.index[[0, 2]], 'B'])

st.write('Example6: index objects')
index6 = pd.Index(['e', 'd', 'a', 'b', 'v'], name='rows')
columns6 = pd.Index(['A', 'B', 'C'], name='cols')
df_index6 = pd.DataFrame(np.random.randn(5, 3), index=index6, columns=columns6)
st.dataframe(df_index6)

st.write('Example7: set index')
col12, col13 = st.columns(2)
col14, col15 = st.columns(2)
df_index7 = pd.DataFrame(
    {
        'a': ['bar', 'bar', 'foo', 'foo'],
        'b': ['one', 'two', 'one', 'two'],
        'c': ['z', 'y', 'x', 'w'],
        'd': [1.0, 2.0, 3.0, 4.0]
    }
)
with col12:
    st.dataframe(df_index7)
with col13:
    index7 = df_index7.set_index('c')
    st.dataframe(index7)
with col14:
    index7_1 = df_index7.set_index(['a', 'b'])
    st.dataframe(index7_1)
with col15:
    index7_2 = ['it', 'will', 'be', 'legendary']
    df_index7_1 = df_index7.copy()
    df_index7_1.index = index7_2
    st.dataframe(df_index7_1)

st.markdown('''#### Multiple indexes  🚻''')
st.write('from_tuples')
arrays1 = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"],
]
tuples1 = list(zip(*arrays1))
index1 = pd.MultiIndex.from_tuples(tuples1, names=["first", "second"])
df_ind1 = pd.DataFrame(np.random.randn(8, 5), index=index1)
st.dataframe(df_ind1)

st.write('from_product')
iterables2 = [["bar", "baz", "foo", "qux"], ["one", "two"]]
index2 = pd.MultiIndex.from_product(iterables2, names=["first", "second"])
df_ind2 = pd.DataFrame(np.random.randn(8, 5), index=index2)
st.dataframe(df_ind2)

st.write('from_frame')
df_tmp1 = pd.DataFrame(
    [["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]],
    columns=["first", "second"],
)
index3 = pd.MultiIndex.from_frame(df_tmp1)
df_ind3 = pd.DataFrame(np.random.randn(4, 5), index=index3)
st.dataframe(df_ind3)

st.write('index = array')
array4 = [
    np.array(["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"]),
    np.array(["one", "two", "one", "two", "one", "two", "one", "two"]),
]
df_ind4 = pd.DataFrame(np.random.randn(8, 4), index=array4)
st.dataframe(df_ind4)

df_ind5 = pd.DataFrame(np.random.randn(3, 8), index=["A", "B", "C"], columns=array4)
st.dataframe(df_ind5)

df_ind5_1 = df_ind5.copy()
df_ind5_1 = df_ind5_1.T
st.dataframe(df_ind5_1)

df_ind6 = pd.DataFrame(np.random.randn(8, 8), index=array4[:6], columns=array4[:6])
st.dataframe(df_ind6)

st.write('series')
s = pd.Series(
    [1, 2, 3, 4, 5, 6],
    index=pd.MultiIndex.from_product([["A", "B"], ["c", "d", "e"]]),
)
df_ind7 = pd.DataFrame(s)
st.dataframe(df_ind7)

st.write('MultiIndex > 2')
array8 = [
    np.array(["beaver", "beaver", "beaver", "beaver", "beaver", "beaver", "beaver", "beaver"]),
    np.array(["masha", "masha", "masha", "masha", "sveta", "sveta", "sveta", "sveta"]),
    np.array(["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"]),
    np.array(["one", "two", "one", "two", "one", "two", "one", "two"])
]
df_ind8 = pd.DataFrame(np.random.randn(8, 4), index=array8)
st.dataframe(df_ind8)
df_ind8_1 = pd.DataFrame(np.random.randn(8, 4), index=array8[1:4])
st.dataframe(df_ind8_1)
df_ind8_2 = pd.DataFrame(np.random.randn(8, 4), index=array8[2:4])
st.dataframe(df_ind8_2)
df_ind8_3 = pd.DataFrame(np.random.randn(8, 4), index=array8[3:4])
st.dataframe(df_ind8_3)

st.write('Using slicers')
def mklbl(prefix, n):
    return ["%s%s" % (prefix, i) for i in range(n)]
miindex = pd.MultiIndex.from_product(
    [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
)
micolumns = pd.MultiIndex.from_tuples(
    [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
)
dfmi = (
    pd.DataFrame(
        np.arange(len(miindex) * len(micolumns)).reshape(
            (len(miindex), len(micolumns))
        ),
        index=miindex,
        columns=micolumns,
    )
    .sort_index()
    .sort_index(axis=1)
)
st.dataframe(dfmi)

st.dataframe(dfmi.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :])

idx = pd.IndexSlice
st.dataframe(dfmi.loc[idx[:, :, ["C1", "C3"]], idx[:, "foo"]])

st.markdown('''#### Index types 🛠''')
st.write('Category index')
df10 = pd.DataFrame({"A": np.arange(6), "B": list("aabbca")})
df10["B"] = df10["B"].astype(pd.api.types.CategoricalDtype(list("cab")))
df10_2 = df10.set_index("B")
st.dataframe(df10_2)

st.write('Interval Index')
df11 = pd.DataFrame(
    {"A": [1, 2, 3, 4]}, index=pd.IntervalIndex.from_breaks([0, 1, 2, 3, 4])
)
st.dataframe(df11)



st.markdown('### Pandas dataframes with missing data')
df = pd.DataFrame(
    np.random.rand(5, 3),
    index=['a', 'c', 'e', 'f', 'h'],
    columns=['one', 'two', 'three']
)
df['four'] = "bar"
df['five'] = df['one'] > 0
df_nan = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])
st.dataframe(df_nan)

st.write('np.nan with float')
df_nan2 = pd.DataFrame(
    {
        "AA": [random.choice([np.nan, np.random.rand()]) for _ in range(10)],
        "BB": [random.choice([np.nan, np.random.rand()]) for _ in range(10)],
        "CC": [random.choice([np.nan, np.random.rand()]) for _ in range(10)],
        "AB": [random.choice([np.nan, np.random.rand()]) for _ in range(10)],
        "BC": [random.choice([np.nan, np.random.rand()]) for _ in range(10)],
        "CA": [random.choice([np.nan, np.random.rand()]) for _ in range(10)],
        "AC": [random.choice([np.nan, np.random.rand()]) for _ in range(10)],
        "BA": [random.choice([np.nan, np.random.rand()]) for _ in range(10)],
        "CB": [random.choice([np.nan, np.random.rand()]) for _ in range(10)]
    }
)
st.dataframe(df_nan2)

st.write('np.nan with int')
df_nan3 = pd.DataFrame(pd.Series([1, 2, np.nan, 4], dtype=pd.Int64Dtype()))
st.dataframe(df_nan3)

st.write('np.nan with datetime')
df_nan4 = df_nan.copy()
df_nan4["timestamp"] = pd.Timestamp("20220315")
df_nan4.loc[['a', 'c', 'h'], ['one', 'timestamp']] = np.nan
st.dataframe(df_nan4)

st.write('None')
df_nan5 = pd.DataFrame(pd.Series(["obladi", "oblada", None, "goes", "on"]))
st.dataframe(df_nan5)

st.write('pd.NA')
df_nan6 = pd.DataFrame(
    {"a": list(range(4)), "b": list("ablm"), "c": ["who", "are", pd.NA, "you"]}
)
st.dataframe(df_nan6)

st.write('from CSV')
csv = st.file_uploader('upload a test csv')
if csv:
    data = pd.read_csv(csv, index_col='index')
    #data = data.drop(data.columns[0], axis=1)
    df_csv = pd.DataFrame(data)
    st.dataframe(df_csv)
    # print(df_csv.dtypes)
    df_csv_dt = df_csv.copy()
    df_csv_dt =df_csv_dt.astype({"name": 'string', "last name": 'string', "DOB": 'datetime64', "ZIP": 'string', "Is in team": 'boolean'})
    st.dataframe(df_csv_dt)
    #print(df_csv_dt.dtypes)



st.markdown('''### Pivot tables''')
df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 6,
        "B": ["A", "B", "C"] * 8,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
        "D": np.random.randn(24),
        "E": np.random.randn(24),
        "F": [datetime.datetime(2013, i, 1) for i in range(1, 13)]
        + [datetime.datetime(2013, i, 15) for i in range(1, 13)],
    }
)
st.dataframe(df)
df_pivot  = pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])
st.dataframe(df_pivot)
df_pivot_2 = pd.pivot_table(df, values="D", index=["B"], columns=["A", "C"], aggfunc=np.sum)
st.dataframe(df_pivot_2)
df_pivot_3 = pd.pivot_table(df, values=["D","E"], index=["B"], columns=["A", "C"], aggfunc=np.sum)
st.dataframe(df_pivot_3)



st.markdown('''### Large table''')
fake = Faker()
Faker.seed(123)
data_fake = []
for i in range(10000):
    profile = fake.profile()
    data_fake.append(
        {
            "name": profile["name"],
            "username": profile["username"],
            "phone": fake.phone_number(),
            "sex": profile["sex"],
            "address": profile["address"],
            "email": profile["mail"],
            "DOB": profile["birthdate"],
            "location": profile["current_location"],
            "job": profile["job"],
            "company": profile["company"],
            "ssn": profile["ssn"],
            "plate": fake.license_plate(),
            "credit card": fake.credit_card_number(),
            "Time zone": fake.timezone()

        }
    )
df_fake = pd.DataFrame(data_fake)
st.dataframe(df_fake)
