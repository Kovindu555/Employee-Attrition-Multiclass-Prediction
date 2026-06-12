from flask import (
    Blueprint, render_template, request, flash, redirect, url_for, session, flash
)
from database import create_user, verify_password

auth = Blueprint('auth', __name__)

# Sign-up
@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        # Validation
        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('sign_up.html')
        
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return render_template('sign_up.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('sign_up.html')
        
        # Create user
        success = create_user(email, password)

        if not success:
            flash('Email already registered.', 'error')
            return render_template('sign_up.html')
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('sign_up.html')


# Sign-in
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', "").strip()
        password = request.form.get('password', "")

        if not verify_password(email, password):
            flash('Invalid email or password.', 'error')
            return render_template('sign_in.html')
        
        # Store user in session
        session.clear()
        session['user_email'] = email.lower().strip()

        # Redirect to home page after successful login
        return redirect(url_for('index'))
    
    return render_template('sign_in.html')

# Logout
@auth.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('auth.login'))