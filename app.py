from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from chatbot import get_chatbot_response  # our FAISS + FAQ chatbot

app = Flask(__name__)
app.secret_key = os.urandom(24)

# -------------------- Database Setup --------------------
instance_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance')
os.makedirs(instance_path, exist_ok=True)

db_path = os.path.join(instance_path, 'database.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------------------- Models --------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    message = db.Column(db.String(1000))
    response = db.Column(db.String(1000))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    welcome_message = "Hi there! I am your MedShop assistant 🤖. How can I help you today?"

    # ✅ Check if welcome message already exists for this user
    existing = ChatHistory.query.filter_by(
        user_id=session['user_id'],
        message="__welcome__"
    ).first()

    # ✅ Only add if not already in chat history
    if existing is None:
        entry = ChatHistory(
            user_id=session['user_id'],
            message="__welcome__",
            response=welcome_message
        )
        db.session.add(entry)
        db.session.commit()

    return render_template('index.html', welcome_message=welcome_message)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            return redirect(url_for('index'))
        else:
            return "Invalid credentials. Go back and try again."
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            return "Username already exists."
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    chats = ChatHistory.query.filter_by(user_id=session['user_id']).all()
    return render_template('history.html', history=chats)

@app.route('/history_json')
def history_json():
    if 'user_id' not in session:
        return jsonify([])

    chats = ChatHistory.query.filter_by(user_id=session['user_id']).order_by(ChatHistory.timestamp).all()
    history = []

    for chat in chats:
        if chat.message == "__welcome__":
            history.append({
                'sender':'bot', 
                'message': chat.response, 
                'timestamp': chat.timestamp.isoformat()  # ISO format
            })
        else:
            history.append({
                'sender':'user', 
                'message': chat.message, 
                'timestamp': chat.timestamp.isoformat()
            })
            history.append({
                'sender':'bot', 
                'message': chat.response, 
                'timestamp': chat.timestamp.isoformat()
            })

    return jsonify(history)


@app.route('/get', methods=['POST'])
def get():
    if 'user_id' not in session:
        return jsonify({'msg': "Please login first."})
    
    user_msg = request.form.get('msg','').strip()
    if not user_msg:
        return jsonify({'msg':"Please type something."})

    # Save user message
    user_entry = ChatHistory(user_id=session['user_id'], message=user_msg, response="")
    db.session.add(user_entry)
    db.session.commit()

    # Get bot response
    response_text = get_chatbot_response(user_msg)

    # Save bot response
    bot_entry = ChatHistory(user_id=session['user_id'], message=user_msg, response=response_text)
    db.session.add(bot_entry)
    db.session.commit()

    return jsonify({
        'user_timestamp': user_entry.timestamp.isoformat(),
        'bot_msg': response_text,
        'bot_timestamp': bot_entry.timestamp.isoformat()
    })

# -------------------- Run App --------------------
if __name__=="__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
