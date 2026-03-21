import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Create database
conn = sqlite3.connect("titanic.db")

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Save to database
df.to_sql("titanic", conn, if_exists="replace", index=False)

# Query: survivors by sex
query1 = '''
SELECT Sex, COUNT(*) as count
FROM titanic
WHERE Survived = 1
GROUP BY Sex
'''
df1 = pd.read_sql_query(query1, conn)

plt.bar(df1['Sex'], df1['count'])
plt.title("Survivors by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.savefig("survivors_by_sex.png")
plt.clf()

# Query: average age by class
query2 = '''
SELECT Pclass, AVG(Age) as avg_age
FROM titanic
GROUP BY Pclass
'''
df2 = pd.read_sql_query(query2, conn)

plt.plot(df2['Pclass'], df2['avg_age'], marker='o')
plt.title("Average Age by Class")
plt.xlabel("Class")
plt.ylabel("Age")
plt.savefig("avg_age_by_class.png")

conn.close()
