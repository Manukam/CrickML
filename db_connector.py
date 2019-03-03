import pymysql.cursors
import numpy as np

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='crickml',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT * FROM `pre_wc_2015`"
        cursor.execute(sql)
        result = cursor.fetchall()
        for player in result:
            # print (player['home_strike_rate'])
            # u

        # print(result)
finally:
    connection.close()