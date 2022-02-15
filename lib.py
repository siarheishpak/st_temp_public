from lib1 import *
import mysql.connector

def recursion_sum(n):
    if n == 0:
        return 0
    else:
        return (n + recursion_sum(n-1))

def unbound_locacl_error():
    a += 1

def plus_one():
    n = initial_data_sum()
    return n+1

def get_mysql_connectiton():
    mysql_connection = mysql.connector.connect(
      host="readonly.alpha.rds.cloud",
      user="user",
      password="qwe123qwe",
      database="qwe123qwe"
    )
    return mysql_connection
