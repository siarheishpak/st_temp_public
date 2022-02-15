import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
# from lib import *

###################################
#
# Base class exceptions
#
###################################
st.header('Base classes')
if st.checkbox('Base exceptioin'):
    base_error_type = st.radio(
        label = 'Type of base error',
        options = ('none', 'with_traceback'))
    if base_error_type == 'none':
        raise BaseException('Manually raised BaseException')
    elif base_error_type == 'with_traceback':
        tb = sys.exc_info()[2]
        raise BaseException(...).with_traceback(tb)

if st.checkbox('Exception'):
    raise Exception('Manually raised Exception')

if st.checkbox('ArithmeticError'):
    raise ArithmeticError('Manually raised ArithmeticError')

if st.checkbox('BufferError'):
    raise BufferError('Manually raised BufferError')

if st.checkbox('LookupError'):
    raise LookupError('Manually raised LookupError')

###################################
#
# Concrete exceptions
#
###################################
st.header('Concrete exceptions')
if st.checkbox('AssertionError'):
    assertion_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if assertion_error == 'raise':
        raise AssertionError('Manually raised AssertionError')
    elif assertion_error == 'generic':
        var1 = 1 # random variable
        assert var1 == 0, 'var1 should be 1'

if st.checkbox('AttributeError'):
    attribute_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if attribute_error == 'raise':
        raise AttributeError('Manually raised AttributeError')
    elif attribute_error == 'generic':
        attribute_string = 'qwe123QWE'
        attribute_string.append()

if st.checkbox('EOFError'):
    raise EOFError('Manually raised EOFError')

if st.checkbox('GeneratorExit'):
    raise GeneratorExit('Manually raised GeneratorExit')

if st.checkbox('ImportError'):
    raise ImportError('Manually raised ImportError')

if st.checkbox('ModuleNotFoundError'):
    raise ModuleNotFoundError('Manually raised ModuleNotFoundError')

if st.checkbox('IndexError'):
    index_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if index_error == 'raise':
        raise IndexError('Manually raised IndexError')
    elif index_error == 'generic':
        index_error_array = [1, 2, 3]
        st.write(index_error_array[3])

if st.checkbox('KeyError'):
    key_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if key_error == 'raise':
        raise KeyError('Manually raised KeyError')
    elif key_error == 'generic':
        index_error_array = {"key1": "value1", "key2":"value2"}
        st.write(index_error_array["key3"])

if st.checkbox('KeyboardInterrupt'):
    raise KeyboardInterrupt('Manually raised KeyboardInterrupt')

if st.checkbox('MemoryError'):
    raise MemoryError('Manually raised MemoryError')

if st.checkbox('NameError'):
    name_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if name_error == 'raise':
        raise NameError('Manually raised NameError')
    elif name_error == 'generic':
        st.write(not_initialized_variable)

if st.checkbox('NotImplementedError'):
    raise NotImplementedError('Manually raised NotImplementedError')

if st.checkbox('OSError'):
    os_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if os_error == 'raise':
        raise OSError('Manually raised OSError')
    elif os_error == 'generic':
        for i in range(7):
            st.write(i)
            st.write(os.ttyname(i))

if st.checkbox('OverflowError'):
    raise OverflowError('Manually raised OverflowError')

if st.checkbox('RecursionError'):
    recursion_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if recursion_error == 'raise':
        raise RecursionError('Manually raised RecursionError')
    elif recursion_error == 'generic':
        st.write(recursion_sum(1001))

if st.checkbox('ReferenceError'):
    raise ReferenceError('Manually raised ReferenceError')

if st.checkbox('RuntimeError'):
    raise RuntimeError('Manually raised RuntimeError')
# for generic flow use lissajous.py script. Comment out fig destroying in anymation loop

if st.checkbox('StopIteration'):
    raise StopIteration('Manually raised StopIteration')

if st.checkbox('StopAsyncIteration'):
    raise StopAsyncIteration('Manually raised StopAsyncIteration')

if st.checkbox('SyntaxError'):
    syntax_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if syntax_error == 'raise':
        raise SyntaxError('Manually raised SyntaxError')
#    elif syntax_error == 'generic':
#        st.write(f'radio box: {'})

if st.checkbox('IndentationError'):
    raise IndentationError('Manually raised IndentationError')

if st.checkbox('SystemExit'):
    sysexit_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if sysexit_error == 'raise':
        raise SystemExit('Manually raised SystemExit')
    elif sysexit_error == 'generic':
        sys.exit([2])

if st.checkbox('TabError'):
    raise TabError('Manually raised TabError')

if st.checkbox('TypeError'):
    type_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if type_error == 'raise':
        raise TypeError('Manually raised TypeError')
    elif type_error == 'generic':
        attribute_error_data = pd.DataFrame(
            np.random.randn(2, 20),
            columnss = ['x', 'y']
        )
        st.line_chart(attribute_error_data)

if st.checkbox('UnboundLocalError'):
    ul_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if ul_error == 'raise':
        raise UnboundLocalError('Manually raised UnboundLocalError')
    elif ul_error == 'generic':
        a = 5
        unbound_locacl_error()
        st.write(a)

if st.checkbox('UnicodeError'):
    raise UnicodeError('Manually raised UnicodeError')

if st.checkbox('UnicodeEncodeError'):
    raise UnicodeEncodeError('Manually raised UnicodeEncodeError')

if st.checkbox('UnicodeDecodeError'):
    raise UnicodeDecodeError('Manually raised UnicodeDecodeError')

if st.checkbox('UnicodeTranslateError'):
    raise UnicodeTranslateError('Manually raised UnicodeTranslateError')

if st.checkbox('ValueError'):
    value_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if value_error == 'raise':
        raise ValueError('Manually raised ValueError')
    elif value_error == 'generic':
        st.line_chart(data=5)

if st.checkbox('ZeroDivisionError'):
    zero_dev_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if zero_dev_error == 'raise':
        raise ZeroDivisionError('Manually raised ZeroDivisionError')
    elif zero_dev_error == 'generic':
        st.write(5/0)

if st.checkbox('EnvironmentError'):
    env_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if env_error == 'raise':
        raise EnvironmentError('Manually raised EnvironmentError')
    elif env_error == 'generic':
        file = open("no_such_file.txt", 'r')

if st.checkbox('IOError'):
    raise IOError('Manually raised IOError')

if st.checkbox('WindowsError'):
    raise WindowsError('Manually raised WindowsError')


###################################
#
# OS exceptions
#
###################################
st.header('OS exceptions')

if st.checkbox('BlockingIOError'):
    raise BlockingIOError('Manually raised BlockingIOError')

if st.checkbox('ChildProcessError'):
    raise ChildProcessError('Manually raised ChildProcessError')

if st.checkbox('ConnectionError'):
    raise ConnectionError('Manually raised ConnectionError')

if st.checkbox('BrokenPipeError'):
    raise BrokenPipeError('Manually raised BrokenPipeError')

if st.checkbox('ConnectionAbortedError'):
    raise ConnectionAbortedError('Manually raised ConnectionAbortedError')

if st.checkbox('ConnectionRefusedError'):
    raise ConnectionRefusedError('Manually raised ConnectionRefusedError')

if st.checkbox('ConnectionResetError'):
    raise ConnectionResetError('Manually raised ConnectionResetError')

if st.checkbox('FileExistsError'):
    fe_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if fe_error == 'raise':
        raise FileExistsError('Manually raised FileExistsError')
    elif fe_error == 'generic':
        os.mkdir('tmp_dir')
        os.mkdir('tmp_dir')

if st.checkbox('FileNotFoundError'):
    fnf_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if fnf_error == 'raise':
        raise FileNotFoundError('Manually raised FileNotFoundError')
    elif fnf_error == 'generic':
        os_error_image = Image.open(r"not_existed_image")
        st.image(os_error_image)

if st.checkbox('InterruptedError'):
    raise InterruptedError('Manually raised InterruptedError')

if st.checkbox('IsADirectoryError'):
    isadir_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if isadir_error == 'raise':
        raise IsADirectoryError('Manually raised IsADirectoryError')
    elif isadir_error == 'generic':
        if st.button('Create dir'):
            os.mkdir('tmp_dir1')
        if st.button('Run error flow'):
            os.remove('tmp_dir1')
        if st.button('Delete dir'):
            os.rmdir('tmp_dir1')

if st.checkbox('NotADirectoryError'):
    notadir_error = st.radio(
        label = 'Method to invoke an error',
        options = ('raise', 'generic')
    )
    if notadir_error == 'raise':
        raise NotADirectoryError('Manually raised NotADirectoryError')
    elif notadir_error == 'generic':
        open("myfile.txt", "w+")
        os.listdir('myfile.txt')

if st.checkbox('PermissionError'):
    raise PermissionError('Manually raised PermissionError')

if st.checkbox('ProcessLookupError'):
    raise ProcessLookupError('Manually raised ProcessLookupError')

if st.checkbox('TimeoutError'):
    raise TimeoutError('Manually raised TimeoutError')

###################################
#
# Warnings
#
###################################
st.header('Warning')

if st.checkbox('Warning'):
    raise Warning('Manually raised Warning')

if st.checkbox('UserWarning'):
    raise UserWarning('Manually raised UserWarning')

if st.checkbox('DeprecationWarning'):
    raise DeprecationWarning('Manually raised DeprecationWarning')

if st.checkbox('PendingDeprecationWarning'):
    raise PendingDeprecationWarning('Manually raised PendingDeprecationWarning')

if st.checkbox('SyntaxWarning'):
    raise SyntaxWarning('Manually raised SyntaxWarning')

if st.checkbox('RuntimeWarning'):
    raise RuntimeWarning('Manually raised RuntimeWarning')

if st.checkbox('FutureWarning'):
    raise FutureWarning('Manually raised FutureWarning')

if st.checkbox('ImportWarning'):
    raise ImportWarning('Manually raised ImportWarning')

if st.checkbox('UnicodeWarning'):
    raise UnicodeWarning('Manually raised UnicodeWarning')

if st.checkbox('EncodingWarning'):
    raise EncodingWarning('Manually raised EncodingWarning')

if st.checkbox('BytesWarning'):
    raise BytesWarning('Manually raised BytesWarning')

if st.checkbox('ResourceWarning'):
    raise ResourceWarning('Manually raised ResourceWarning')

###################################
#
# Nested Errors
#
###################################
st.header('Nested Errors')
if st.checkbox('Nested Errors'):
    a = plus_one()
    st.write(a)


if st.checkbox('misc'):
    connection = get_mysql_connectiton()
