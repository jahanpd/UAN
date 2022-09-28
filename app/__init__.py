# https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

from flask import Flask
from config import Config
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
import torch
import pyrebase
import json
from app.APN.model.Network import network

# initialize app
app = Flask(__name__)
app.config.from_object(Config)

# initialize login manager
login_manager = LoginManager(app)
login_manager.login_view = '/login'
# initialize firebase authentication
firebase = pyrebase.initialize_app(app.config['FIREBASE_CONFIG'])
auth = firebase.auth()

# Add bootstrap to app
Bootstrap(app)

# load data dicts
print(__name__)
with open('./app/static/feature_list.json', 'r') as file:
    data_dict = json.load(file)
file.close()

with open('./app/static/outcome_data.json', 'r') as file:
    outcome_dict = json.load(file)
file.close()

# load model params
state_dict = torch.load(
    './app/APN/model/saved/final-PN.pt', map_location="cpu")

# initialize models
model = network(
    41,
    8,
    16,
)

# load machine learning model
try:
    model.load_state_dict(state_dict)
    print("loaded")
except Exception as e:
    print("couldnt load model")
    print(e)

model.eval()

# import blueprints
from .views.home import home
from .views.login import login_b
from .views.data_entry import data_entry
from .views.display import display
from .views.signup import signup
from .views.password_reset import password_reset

# register blueprints
app.register_blueprint(home)
app.register_blueprint(login_b)
app.register_blueprint(data_entry)
app.register_blueprint(display)
app.register_blueprint(signup)
app.register_blueprint(password_reset)

# login_manager.blueprint_login_views = {
#     'login': '/login',
# }
