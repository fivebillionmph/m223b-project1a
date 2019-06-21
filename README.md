main scripts:

- src/roc.R -- for generating ROC curves and getting AUC
- run-flask.sh -- for running the flask server for annotating.  Assumes a directory exists in data/NeedleImages that contains all the images and a sqlite database exists at src/images.db
- src/trainer.py -- main script for training training neural networks
- src/test-annotated.py -- script for testing neural networks
