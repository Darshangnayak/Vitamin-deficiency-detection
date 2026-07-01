from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from database import get_db_connection, init_db
from models import DISEASE_INFO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
import json
import numpy as np



model = load_model('vitamindeficiency.h5')

# Load class labels
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# Image input size for ResNet
img_height, img_width = 224, 224

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Use ResNet-specific preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict and display results
def predict_image(img_path, true_label=None):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = index_to_class[predicted_index].rstrip()
    confidence = predictions[0][predicted_index] * 100

    print(f"\n🧠 Predicted Label: {predicted_label}")
    print(f"✅ Confidence: {confidence:.2f}%")
    return (predicted_label,confidence)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database
init_db()

# Initialize model (replace with your actual model path)
#classifier = DiseaseClassifier('path/to/your/model.pth')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('profile'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['full_name']
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?', 
            (username, email)
        ).fetchone()
        
        if user:
            flash('Username or email already exists!')
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        conn.execute(
            'INSERT INTO users (username, email, password, full_name) VALUES (?, ?, ?, ?)',
            (username, email, hashed_password, full_name)
        )
        conn.commit()
        conn.close()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            flash('Login successful!')
            return redirect(url_for('profile'))
        else:
            flash('Invalid username or password!')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out!')
    return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC LIMIT 10',
        (session['user_id'],)
    ).fetchall()
    conn.close()
    
    return render_template('profile.html', predictions=predictions)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            
            # Make prediction
            try:
                #predicted_disease, confidence = classifier.predict(filepath)
                predicted_disease,confidence=predict_image(filepath)
                # Save prediction to database
                conn = get_db_connection()
                conn.execute(
                    'INSERT INTO predictions (user_id, image_path, predicted_disease, confidence) VALUES (?, ?, ?, ?)',
                    (session['user_id'], filename, predicted_disease, confidence)
                )
                conn.commit()
                conn.close()
                
                # Get disease information
                disease_info = DISEASE_INFO.get(predicted_disease, {})
                
                return render_template('result.html', 
                                    predicted_disease=predicted_disease,
                                    confidence=confidence,
                                    disease_info=disease_info,
                                    image_url=url_for('static', filename='uploads/' + filename))
            
            except Exception as e:
                flash(f'Error during prediction: {str(e)}')
                return redirect(request.url)
        
        else:
            flash('Invalid file type! Please upload an image.')
    
    return render_template('upload.html')

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)
