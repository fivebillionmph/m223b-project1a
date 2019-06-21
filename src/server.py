import os
import sqlite3
from mod.helper import sqlite_dict_factory
from flask import Flask, render_template, jsonify, send_file, request
app = Flask(__name__, template_folder="server/templates", static_url_path="/static", static_folder="server/static")
app.config["TEMPLATES_AUTO_RELOAD"] = True

FILEDIR = "../data/Needleimages"

def getConnection():
	cxn = sqlite3.connect("images.db")
	cxn.row_factory = sqlite_dict_factory
	return cxn

@app.route("/api/images", methods=["GET"])
def api_images():
	cxn = getConnection()
	c = cxn.cursor()
	c.execute("select name, has_feature, trainable, feature_mask from images")
	res = c.fetchall()
	cxn.close()
	return jsonify(res)

@app.route("/api/save_mask", methods=["POST"])
def api_save_mask():
	cxn = getConnection()
	content = request.json
	name = content["name"]
	mask = content["feature_mask"]
	c = cxn.cursor()
	c.execute("update images set feature_mask = ? where name = ?", (mask, name))
	cxn.commit()
	cxn.close()
	return jsonify(True)

@app.route("/api/set_trainable", methods=["POST"])
def api_set_trainable():
	cxn = getConnection()
	content = request.json
	name = content["name"]
	trainable = 1 if content["trainable"] == 1 else 0
	c = cxn.cursor()
	c.execute("update images set trainable = ? where name = ?", (trainable, name))
	cxn.commit()
	cxn.close()
	return jsonify(True)

@app.route("/image/<name>")
def get_image(name):
	return send_file(os.path.join(FILEDIR, name))

@app.route("/annotator")
def annotator():
	return render_template("annotator.html")

@app.route("/trainer")
def trainer():
	return render_template("trainer.html")

@app.route("/")
def index():
	return render_template("index.html")
