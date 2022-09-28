from flask import render_template, flash, redirect, url_for, session, request
from flask_login import current_user, login_user, logout_user, login_required
from app import app
from app.forms import LoginForm, DataInput
from app import User
from app import auth, model, data_dict, outcome_dict

from .plotting import plot_heatmap, plot_selfattention
from bokeh.resources import INLINE

import torch
from scipy.stats import beta
import numpy as np
import json

js_resources = INLINE.render_js()
css_resources = INLINE.render_css()


# @app.route('/')
# @app.route('/index')
# def index():
#     # perform logic here eg
#     user = {'username': 'ANZSCTS'}
#     info = [{'role': 'Auditing', 'yolo': 'swagger'}]
#     return render_template('index.html', user=user, info=info)


# @app.route('/input_attention', methods=['GET', 'POST'])
# @login_required
# def input_attention():
#     inputform = DataInput()
#     types = {}
#     for feat in data_dict:
#         types[feat["colname"]] = feat["class"]
#     print("data_dict", len(data_dict))
#     if inputform.validate_on_submit():
#         data = {}
#         for f in inputform:
#             data[f.id] = f.data
#         dict_ = {}
#         for key in types:
#             dict_[key] = float(data[key])
#         _ = session.pop('patient', None)
#         session['patient'] = dict_
#         print(dict_)
#         # data = dict(request.form)
#         # if '' in data.values():
#         #     flash('Please input valid data for all fields')
#         # elif len(data) == 0:
#         #     flash('Please input some data')
#         # elif ' ' not in data.values():
#         #     _ = session.pop('patient', None)
#         #     session['patient'] = data
#         #     flash('Please select an outcome')
#         # else:
#         #     _ = session.pop('patient', None)
#         #     session['patient'] = data

#         return redirect(url_for('display_attention'))

#     patient = session.get('patient')

#     return render_template(
#         'input_attention2.html',
#         form=inputform,
#         patient=patient,
#         data_dict=types)


@app.route('/get_patient')
@login_required
def get_patient():
    prefill = session.get('patient')
    return prefill


@app.route('/get_prediction')
@login_required
def get_prediction():
    prefill = session.get('prediction')
    return prefill


@app.route('/display_attention')
@login_required
def display_attention():
    data = session.get('patient')
    row = np.array([data[d['colname']] for d in data_dict])
    mask = row != -1
    M = 20
    X_ = torch.tensor(row[np.newaxis, ...]).repeat(M, 1)
    model.sample = True
    single_prob, probs, N, tup = model(xv=X_.to(torch.float32))
    attn, attn_x, beta_params = tup
    # print(x_beta_params)
    # print(beta_params)
    prior_params = N.double().cpu().detach().numpy()
    post_params = beta_params.double().cpu().detach().numpy()
    prior_dist = beta(prior_params[0, :, 0], prior_params[0, :, 1])
    post_dist = beta(post_params[0, :, 0], post_params[0, :, 1])
    probs = probs.mean(0).double().cpu().detach().numpy()[:, mask] * 100

    p = single_prob.mean(0).double().cpu().detach().numpy() * 100
    lb = post_dist.ppf(0.025) * 100
    ub = post_dist.ppf(0.975) * 100

    baseline = prior_dist.mean() * 100
    lb_ = prior_dist.ppf(0.025) * 100
    ub_ = prior_dist.ppf(0.975) * 100
    # process prob and credible intervals to dictionary
    output = {}
    ycolname = np.array([
        'Readmission to ICU', 'Reintubation', 'Return to Theatre',
        'Postoperative Kidney Injury', 'Postoperatve Haemofiltration',
        'New Postoperative Arrhythmia', 'Deep Sternal Wound Infection',
        '30 Day Mortality'
        ])
    xcolname = np.array([d['name'] for d in data_dict])
    subset = xcolname[mask]

    for i, name in enumerate(ycolname):
        pred = [lb[i], p[i], ub[i]]
        base = [lb_[i], baseline[i], ub_[i]]
        output[name] = {
            'prediction': pred,
            'baseline': base
        }

    output_json = json.dumps(output)
    # direction
    direction = np.sign(probs - (baseline.reshape(-1, 1)))
    attn = attn.mean(0).cpu().detach().numpy()[:, mask] * direction
    attn_x = attn_x.mean(0).cpu().detach().numpy()[:, mask]

    script_hm, div_hm = plot_heatmap(
        attn, probs, p, row[mask], subset, ycolname
        )

    script_hm_x, div_hm_x = plot_selfattention(
        attn_x[mask, :], subset
        )

    html = render_template(
        'display_attention.html',
        plot_script_hm=script_hm,
        plot_div_hm=div_hm,
        plot_script_hm_x=script_hm_x,
        plot_div_hm_x=div_hm_x,
        js_resources=js_resources,
        css_resources=css_resources,
        output_json=output_json
        )

    return html


@app.route('/input_bayes_net', methods=['GET', 'POST'])
@login_required
def input_bayes_net():
    return "not implemented"


@app.route('/attention_api', methods=['GET'])
@login_required
def attention_call():
    data = request.args.get("data")
    if data:
        x = json.loads(data)  # should be dict of form { "x_vals":(B, F, 1), "x_cols":(B,F), "y_cols":(B,F) }
        x['x_vals'] = np.array(x['x_vals']).astype(float)
        x['x_cols'] = np.array(x['x_cols']).astype(int)
        x['y_cols'] = np.array(x['y_cols']).astype(int)

        LOGITS, ALPHA, attention_weights = model(x)  # (B, O, 1), (B,H,O,F)
        y = {}
        y['logits'] = LOGITS.numpy().tolist()
        y['alpha'] = ALPHA.numpy().tolist()
        y['attn'] = attention_weights.numpy().tolist()

        return json.dumps(y)

    return "failed"


# these are view functions
# @app.route('/login', methods=['GET', 'POST'])  # GET request returns info to client, POST returns info to server
# def login():
#     if current_user.is_authenticated:
#         return redirect(url_for('index'))

#     form = LoginForm()
#     if form.validate_on_submit():
#         email = form.username.data
#         password = form.password.data
#         try:
#             u = auth.sign_in_with_email_and_password(email, password)
#             userID = u['localId']
#             refreshToken = u['refreshToken']
#             idToken = u['idToken']

#             user = User(userID, idToken, refreshToken)
#             login_user(user, remember=form.remember_me.data)

#             flash('Login requested for user {}, remember_me={}'.format(
#                 form.username.data, form.remember_me.data))

#             return redirect(url_for('index'))
#         except Exception:
#             flash('Invalid username or password')
#             return redirect(url_for('login'))

#     return render_template('login.html', title='Sign In', form=form)


# @app.route('/logout')
# def logout():
#     logout_user()
#     return redirect(url_for('index'))
