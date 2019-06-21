import sqlite3
from mod.helper import sqlite_dict_factory
import sys
import os
import csv

conn = sqlite3.connect("images.db")
conn.row_factory = sqlite_dict_factory
c = conn.cursor()

images_dir = sys.argv[1]

if len(sys.argv) > 2 and sys.argv[2] == "init":
	c.execute("""
		create table images (
			id integer primary key,
			name text not null unique,
			has_feature integer not null,
			trainable integer not null,
			feature_mask text
		)
	""")

labels_filename = os.path.join(images_dir, "Labels.csv")
with open(labels_filename) as f:
	reader = csv.reader(f)
	next(reader) # throw away first line
	for line in reader:
		name = line[1]
		has_feature = 1 if line[2] == "yes" else 0
		c.execute("select id from images where name = ?", (name, ))
		res = c.fetchone()
		if res is None:
			c.execute("insert into images values(NULL, ?, ?, ?, NULL)", (name, has_feature, 0))
		else:
			c.execute("update images set has_feature = ? where id = ?", (has_feature, res["id"]))

conn.commit()
conn.close()
