from flask import Blueprint
from flask import render_template, flash, redirect, url_for
from app.forms.login import PasswordForm
from app import auth


password_reset = Blueprint('password_reset', __name__,
                           template_folder='templates',
                           static_folder='static')


# GET request returns info to client, POST returns info to server
@password_reset.route('/password_reset', methods=['GET', 'POST'])
def password_reset_route():
    form = PasswordForm()
    if form.validate_on_submit():
        email = form.email.data
        try:
            auth.send_password_reset_email(email)
            flash('Password reset email sent. Check email inbox.')
            return redirect(url_for('home.index'))
        except Exception:
            flash('Passord Reset Failed. Check email entered')
            return redirect(url_for('login.login_route'))

    return render_template('password_reset.html', title='Sign In', form=form)
