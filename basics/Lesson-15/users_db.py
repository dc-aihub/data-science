import sqlite3
import json
import csv
from datetime import datetime

db_name = "test"
sql_transaction = []

connection = sqlite3.connect('{}.db'.format(db_name))
c = connection.cursor()

def create_table():
	c.execute("""CREATE TABLE IF NOT EXISTS users
			(id TEXT PRIMARY KEY,
			first_name TEXT, 
			last_name TEXT, 
			email TEXT, 
			gender TEXT)""")

def sql_insert_user(id,first_name,last_name,email,gender):
	try:

		sql = """INSERT INTO users (id, first_name, last_name, email, gender) VALUES ("{}","{}","{}","{}","{}");""".format(id, first_name, last_name, email, gender)
		print(sql)
		try:
			c.execute(sql)
			connection.commit()
			# print("successfully inserted", id)
		except:
			print("Something went wrong with entering", id)
			pass

	except Exception as e:
		print('insertion',str(e))


if __name__ == "__main__":
	create_table()

	# sql = "SELECT * FROM users;"
	# c.execute(sql)
	# result = c.fetchall()

	# print(result)

	with open("./users.csv", buffering=1000) as f:
		users = csv.reader(f, delimiter=',')
		for row in users:
			sql_insert_user(row[0], row[1], row[2], row[3], row[4])