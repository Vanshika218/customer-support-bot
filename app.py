from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from chatbot import get_chatbot_response
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# -----------------------------
# Database Setup
# -----------------------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    message = db.Column(db.String(1000))
    response = db.Column(db.String(1000))

db.create_all()

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('chat'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get():
    if 'user_id' not in session:
        return jsonify({'msg': "Please login first."})
    user_msg = request.form['msg']
    bot_reply = get_chatbot_response(user_msg)

    # Save to history
    history_entry = ChatHistory(user_id=session['user_id'], message=user_msg, response=bot_reply)
    db.session.add(history_entry)
    db.session.commit()

    return jsonify({'msg': bot_reply})

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    chats = ChatHistory.query.filter_by(user_id=session['user_id']).all()
    return render_template('history.html', chats=chats)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
