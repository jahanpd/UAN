from flask import Blueprint, render_template, session, url_for, redirect
from flask_login import login_required
from app.forms.attention import DataInput
from app import data_dict


data_entry = Blueprint('data_entry', __name__,
                       template_folder='templates',
                       static_folder='static')


@data_entry.route('/input_attention', methods=['GET', 'POST'])
@login_required
def input_attention():
    inputform = DataInput()
    types = {}
    for feat in data_dict:
        types[feat["colname"]] = feat["class"]
    print("data_dict", len(data_dict))
    if inputform.validate_on_submit():
        data = {}
        for f in inputform:
            data[f.id] = f.data
        dict_ = {}
        for key in types:
            dict_[key] = float(data[key])
        _ = session.pop('patient', None)
        session['patient'] = dict_
        print(dict_)
        # data = dict(request.form)
        # if '' in data.values():
        #     flash('Please input valid data for all fields')
        # elif len(data) == 0:
        #     flash('Please input some data')
        # elif ' ' not in data.values():
        #     _ = session.pop('patient', None)
        #     session['patient'] = data
        #     flash('Please select an outcome')
        # else:
        #     _ = session.pop('patient', None)
        #     session['patient'] = data

        return redirect(url_for('display.display_attention'))

    patient = session.get('patient')

    return render_template(
        'input_attention2.html',
        form=inputform,
        patient=patient,
        data_dict=types)
