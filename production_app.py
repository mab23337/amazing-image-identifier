"""
Amazing Image Identifier - Raspberry Pi Optimized Version
No EasyOCR to prevent freezing on Pi
"""

import os
# Fix OpenBLAS warning
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import io
import base64
import json
import time
import logging
from datetime import datetime
import hashlib
import sqlite3
from PIL import Image, ImageDraw
import colorsys
from collections import Counter
import re

# Import ML libraries
try:
    import cv2
    import numpy as np
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("WARNING: ML libraries not installed.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
#SR-47 Limit
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Enable CORS
CORS(app)

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('history', exist_ok=True)

# Initialize database
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT NOT NULL,
                  upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  processing_time REAL,
                  caption TEXT,
                  objects_detected TEXT,
                  colors TEXT,
                  has_text BOOLEAN,
                  file_hash TEXT UNIQUE)''')
    conn.commit()
    conn.close()
    logger.info("Database initialized")

init_db()

# Initialize ML models (lazy loading)
models = {
    'captioning': None,
    'object_detection': None
}

def load_models():
    """Load ML models on first use"""
    global models
    if not HAS_ML:
        logger.warning("ML libraries not available")
        return
    
    try:
        if models['captioning'] is None:
            logger.info("Loading BLIP captioning model...")
            models['captioning'] = {
                'processor': BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
                'model': BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            }
            logger.info("BLIP model loaded")
        
        if models['object_detection'] is None:
            logger.info("Loading DETR object detection model...")
            models['object_detection'] = {
                'processor': DetrImageProcessor.from_pretrained("facebook/detr-resnet-50"),
                'model': DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            }
            logger.info("DETR model loaded")
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image_file(file_path):
    """Validate that file is actually an image"""
    try:
        Image.open(file_path).verify()
        return True
    except:
        return False
  #SR-48      
def validate_magic_bytes(file):
    """Validate image by checking magic header bytes (SR-48)"""
    header = file.read(8)
    file.seek(0) # Reset file pointer after reading
    
    # JPEG
    if header.startswith(b'\xff\xd8\xff'):
        return True
        
    # PNG
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        return True
        
        return False
        
#SR-46 
def sanitize_filename(filename):
    """Sanitize filename to prevent directory traversal"""
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\-.]', '_', filename)
    return secure_filename(filename)

def get_file_hash(file_path):
    """Generate SHA256 hash of file"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_dominant_colors(image_path, num_colors=5):
    """Extract dominant colors from image"""
    try:
        img = Image.open(image_path)
        img = img.resize((150, 150))
        img = img.convert('RGB')
        
        pixels = list(img.getdata())
        color_counts = Counter(pixels)
        dominant = color_counts.most_common(num_colors)
        
        color_names = []
        for color, _ in dominant:
            r, g, b = color
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            if v < 0.2:
                name = "black"
            elif s < 0.1:
                name = "gray" if v < 0.9 else "white"
            elif h < 0.05 or h > 0.95:
                name = "red"
            elif h < 0.15:
                name = "orange"
            elif h < 0.25:
                name = "yellow"
            elif h < 0.45:
                name = "green"
            elif h < 0.55:
                name = "cyan"
            elif h < 0.7:
                name = "blue"
            elif h < 0.85:
                name = "purple"
            else:
                name = "pink"
            
            if name not in color_names:
                color_names.append(name)
        
        return color_names[:5]
    except Exception as e:
        logger.error(f"Error extracting colors: {e}")
        return []

def generate_caption(image_path):
    """Generate natural language caption"""
    if not HAS_ML or models['captioning'] is None:
        return "AI model loading. Please try again."
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        processor = models['captioning']['processor']
        model = models['captioning']['model']
        
        inputs = processor(image, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return "Could not generate caption for this image."

def detect_objects(image_path, confidence_threshold=0.7):
    """Detect objects in image with bounding boxes"""
    if not HAS_ML or models['object_detection'] is None:
        return []
    
    try:
        image = Image.open(image_path).convert('RGB')
        
        processor = models['object_detection']['processor']
        model = models['object_detection']['model']
        
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        # Convert outputs to COCO API format
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i) for i in box.tolist()]
            detected_objects.append({
                'label': model.config.id2label[label.item()],
                'confidence': round(score.item(), 2),
                'box': box
            })
        
        # Ensure at least 5 objects if possible
        if len(detected_objects) < 5 and confidence_threshold > 0.3:
            return detect_objects(image_path, confidence_threshold - 0.1)
        
        return detected_objects[:10]
        
    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        return []

def perform_ocr(image_path):
    """OCR disabled for Raspberry Pi performance"""
    # EasyOCR causes freezing on Raspberry Pi
    # Returning empty result - OCR feature disabled
    return {"has_text": False, "text": "OCR disabled on Raspberry Pi"}

def draw_bounding_boxes(image_path, objects):
    """Draw bounding boxes on image"""
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Color coding by category
        category_colors = {
            'person': '#3498db',
            'vehicle': '#e74c3c',
            'animal': '#2ecc71',
            'furniture': '#f39c12',
            'electronics': '#9b59b6',
            'default': '#95a5a6'
        }
        
        for obj in objects:
            box = obj['box']
            label = obj['label']
            confidence = obj['confidence']
            
            # Determine color
            color = category_colors.get('default')
            for category in category_colors:
                if category in label.lower():
                    color = category_colors[category]
                    break
            
            # Draw rectangle
            draw.rectangle(box, outline=color, width=3)
            
            # Draw label
            text = f"{label} {confidence*100:.0f}%"
            draw.text((box[0], max(0, box[1]-20)), text, fill=color)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return base64.b64encode(img_byte_arr).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error drawing bounding boxes: {e}")
        return None

def save_to_history(filename, processing_time, caption, objects, colors, has_text, file_hash):
    """Save processed image data to database"""
    try:
        conn = sqlite3.connect('images.db')
        c = conn.cursor()
        
        c.execute('''INSERT OR REPLACE INTO images 
                     (filename, processing_time, caption, objects_detected, colors, has_text, file_hash)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (filename, processing_time, caption, 
                   json.dumps(objects), json.dumps(colors), has_text, file_hash))
        
        conn.commit()
        conn.close()
        logger.info(f"Saved to history: {filename}")
    except Exception as e:
        logger.error(f"Error saving to history: {e}")

def get_history(limit=10):
    """Retrieve processing history"""
    try:
        conn = sqlite3.connect('images.db')
        c = conn.cursor()
        
        c.execute('''SELECT filename, upload_time, processing_time, caption, objects_detected, colors
                     FROM images ORDER BY upload_time DESC LIMIT ?''', (limit,))
        
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'filename': row[0],
                'upload_time': row[1],
                'processing_time': row[2],
                'caption': row[3],
                'objects': json.loads(row[4]) if row[4] else [],
                'colors': json.loads(row[5]) if row[5] else []
            })
        
        return history
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return []

def clear_history():
    """Clear all processing history"""
    try:
        conn = sqlite3.connect('images.db')
        c = conn.cursor()
        c.execute('DELETE FROM images')
        conn.commit()
        conn.close()
        logger.info("History cleared")
        return True
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return False

# Routes

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')
    
#SR-48 header validation
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and processing"""
    start_time = time.time()
    
    if 'file' not in request.files:
        logger.warning("No file in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only JPG and PNG are allowed.'}), 400
        
        # SR-48 Magic Byte Check
    if not validate_magic_bytes(file):
        logger.warning("Invalid magic bytes detected")
        return jsonify({'error': 'Invalid image header'}), 400
        
    try:
        #SR-46
        filename = sanitize_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        if not validate_image_file(filepath):
            os.remove(filepath)
            return jsonify({'error': 'File is not a valid image'}), 400
        
        # Load models if not already loaded
        load_models()
        
        file_hash = get_file_hash(filepath)
        
        # Process image
        caption = generate_caption(filepath)
        objects = detect_objects(filepath)
        colors = extract_dominant_colors(filepath)
        ocr_result = perform_ocr(filepath)
        
        # Draw bounding boxes
        annotated_image = draw_bounding_boxes(filepath, objects)
        
        # Read original image
        with open(filepath, 'rb') as img_file:
            original_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Save to history
        save_to_history(filename, processing_time, caption, objects, colors, 
                       ocr_result['has_text'], file_hash)
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'caption': caption,
            'objects': objects,
            'object_count': len(objects),
            'colors': colors,
            'ocr': ocr_result,
            'processing_time': round(processing_time, 2),
            'original_image': f"data:image/jpeg;base64,{original_image}",
            'annotated_image': f"data:image/png;base64,{annotated_image}" if annotated_image else None
        }
        
        logger.info(f"Processed {filename} in {processing_time:.2f}s")
        #CR-49 Priv Scrubbing
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Deleted uploaded file: {filename}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
            logger.info("Deleted file after failure")
            
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<format>', methods=['POST'])
def download_results(format):
    """Download description as .txt or .json"""
    try:
        data = request.json
        
        if format == 'txt':
            content = f"Image Analysis Results\n"
            content += f"=====================\n\n"
            content += f"Caption: {data.get('caption', 'N/A')}\n\n"
            content += f"Objects Detected ({data.get('object_count', 0)}):\n"
            for obj in data.get('objects', []):
                content += f"  - {obj['label']}: {obj['confidence']*100:.1f}%\n"
            content += f"\nColors: {', '.join(data.get('colors', []))}\n"
            content += f"\nProcessing Time: {data.get('processing_time', 0)}s\n"
            
            return send_file(
                io.BytesIO(content.encode()),
                mimetype='text/plain',
                as_attachment=True,
                download_name='analysis.txt'
            )
        
        elif format == 'json':
            return send_file(
                io.BytesIO(json.dumps(data, indent=2).encode()),
                mimetype='application/json',
                as_attachment=True,
                download_name='analysis.json'
            )
        
        else:
            return jsonify({'error': 'Invalid format'}), 400
            
    except Exception as e:
        logger.error(f"Error downloading results: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """Get processing history"""
    try:
        limit = request.args.get('limit', 10, type=int)
        history_data = get_history(limit)
        return jsonify({'success': True, 'history': history_data})
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history/clear', methods=['POST'])
def clear_history_route():
    """Clear processing history"""
    try:
        if clear_history():
            return jsonify({'success': True, 'message': 'History cleared'})
        else:
            return jsonify({'error': 'Failed to clear history'}), 500
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ml_enabled': HAS_ML,
        'models_loaded': all(m is not None for m in models.values()) if HAS_ML else False,
        'ocr_enabled': False  # Disabled for Pi performance
    })

#CR-50
@app.route('/credits')
def credits():
    """Open source credits page"""
    return render_template('credits.html')
    
#SR-47 limit rejection
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File is too large. Maximum size is 10MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Amazing Image Identifier - Raspberry Pi Optimized")
    print("=" * 50)
    print(f"ML Libraries: {'Available' if HAS_ML else 'Not Available'}")
    print(f"OCR: Disabled (for Pi performance)")
    print("Starting server on http://0.0.0.0:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
