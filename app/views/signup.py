from flask import Blueprint
from flask import render_template, flash, redirect, url_for
from app.forms.login import RegistrationForm
from app import auth


signup = Blueprint('signup', __name__,
                   template_folder='templates',
                   static_folder='static')


# GET request returns info to client, POST returns info to server
@signup.route('/signup', methods=['GET', 'POST'])
def signup_route():
    form = RegistrationForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        try:
            auth.create_user_with_email_and_password(email, password)
            flash('Login available now. Check email for validation')
            return redirect(url_for('login.login_route'))
        except Exception:
            flash('Email already in use')
            return redirect(url_for('home.index'))

    return render_template('signup.html', title='Sign Up!', form=form)
