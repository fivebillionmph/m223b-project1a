import sqlite3
from mod.helper import sqlite_dict_factory
import json

def main():
	cxn = sqlite3.connect("images.db")
	cxn.row_factory = sqlite_dict_factory

	cur = cxn.cursor()
	cur.execute("select name, feature_mask from images")
	res = cur.fetchall()
	masks = []
	for row in res:
		if row["feature_mask"] is not None and row["feature_mask"] != "":
			raw_mask = json.loads(row["feature_mask"])
			mask = []
			for point_name in raw_mask["points"]:
				point = raw_mask["points"][point_name]
				mask.append((int(point[0]), int(point[1])))
			masks.append({
				"name": row["name"],
				"mask": mask,
			})

	print(json.dumps(masks))

main()
