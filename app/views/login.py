from flask import Blueprint
from flask import render_template, flash, redirect, url_for
from flask_login import current_user, login_user, logout_user
from app.forms.login import LoginForm
from app.users.users import User
from app import auth


login_b = Blueprint('login', __name__,
                    template_folder='templates',
                    static_folder='static')


# GET request returns info to client, POST returns info to server
@login_b.route('/login', methods=['GET', 'POST'])
def login_route():
    if current_user.is_authenticated:
        return redirect(url_for('home.index'))

    form = LoginForm()
    if form.validate_on_submit():
        email = form.username.data
        password = form.password.data
        try:
            u = auth.sign_in_with_email_and_password(email, password)
            userID = u['localId']
            refreshToken = u['refreshToken']
            idToken = u['idToken']

            user = User(userID, idToken, refreshToken)
            login_user(user, remember=form.remember_me.data)

            flash('Login requested for user {}, remember_me={}'.format(
                form.username.data, form.remember_me.data))

            return redirect(url_for('home.index'))
        except Exception:
            flash('Invalid username or password')
            return redirect(url_for('login.login_route'))

    return render_template('login.html', title='Sign In', form=form)


@login_b.route('/logout')
def logout_route():
    logout_user()
    return redirect(url_for('home.index'))
