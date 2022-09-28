import torch
import numpy as np
import json
from flask import Blueprint, render_template, session
from flask_login import login_required
from app import data_dict, model
from app.plots.plotting import plot_heatmap, plot_selfattention
from bokeh.resources import INLINE
from scipy.stats import beta

js_resources = INLINE.render_js()
css_resources = INLINE.render_css()


display = Blueprint(
    'display', __name__,
    template_folder='templates',
    static_folder='static')


def mapping(x, b):
    if x < 50:
        return (x / (50)) * b
    else:
        return (((x - 50) / (50)) * (100 - b)) + b


@display.route('/display_attention', methods=['GET', 'POST'])
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

    for i in range(8):
        p[i] = mapping(p[i], baseline[i])
        lb[i] = mapping(lb[i], baseline[i])
        ub[i] = mapping(ub[i], baseline[i])

    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            probs[i, j] = mapping(probs[i, j], baseline[i])

    # process prob and credible intervals to dictionary
    output = {}
    ycolname = np.array([
        'Readmission to ICU', 'Reintubation', 'Return to Theatre',
        'Postoperative Kidney Injury', 'Postoperative Haemofiltration',
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
    # direction = np.sign(probs - 50)
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
