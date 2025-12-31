import os
import io
import random
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file
from rembg import remove

app = Flask(__name__)

# ================== CONFIG ==================

UPLOAD_FOLDER = 'uploads'
TEMPLATE_FOLDER = 'templates'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


# ================== TEMPLATE HELPERS ==================

def get_available_templates():
    templates = []
    if os.path.exists(TEMPLATE_FOLDER):
        for filename in os.listdir(TEMPLATE_FOLDER):
            ext = os.path.splitext(filename)[1].lower()
            if ext in ALLOWED_EXTENSIONS:
                templates.append(filename)
    return sorted(templates)


def get_template_path(template_name=None):
    templates = get_available_templates()
    if not templates:
        return None
    if template_name:
        if template_name in templates:
            return os.path.join(TEMPLATE_FOLDER, template_name)
        for ext in ALLOWED_EXTENSIONS:
            full_name = f"{template_name}{ext}"
            if full_name in templates:
                return os.path.join(TEMPLATE_FOLDER, full_name)
        return None
    return os.path.join(TEMPLATE_FOLDER, random.choice(templates))


# ================== BACKGROUND REMOVAL (NO MASK) ==================

def remove_background_rembg(image):
    """
    Use rembg library - no custom mask, no U2-Net handling
    Just pure removal with the library handling everything
    """
    # Convert PIL to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Remove background using rembg
    output = remove(img_byte_arr.read())
    
    # Convert bytes back to PIL Image
    result = Image.open(io.BytesIO(output))
    
    return result


# ================== TEMPLATE PLACEMENT ==================

def create_default_template(size=(1080, 1920)):
    width, height = size
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / height
        r = int(135 + ratio * 30)
        g = int(206 + ratio * 20)
        b = int(235 + ratio * 10)
        gradient[y, :] = [min(r, 255), min(g, 255), min(b, 255)]
    return Image.fromarray(gradient)


def place_rgba_on_template(fg_rgba,
                           template_path=None,
                           position='bottom-center',
                           scale=0.8,
                           padding=0.05):
    """
    Places RGBA cutout on template
    """
    if template_path and os.path.exists(template_path):
        template = Image.open(template_path).convert('RGBA')
        print("Using template:", template_path)
    else:
        template = create_default_template().convert('RGBA')
        print("Using default gradient template")

    template_width, template_height = template.size

    # Get bounding box from alpha channel
    fg_arr = np.array(fg_rgba)
    alpha = fg_arr[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        return template.convert('RGB')

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    fg_cropped = fg_rgba.crop((cmin, rmin, cmax + 1, rmax + 1))

    # Scale
    target_height = int(template_height * scale)
    aspect_ratio = fg_cropped.width / fg_cropped.height
    target_width = int(target_height * aspect_ratio)

    max_width = int(template_width * (1 - 2 * padding))
    if target_width > max_width:
        target_width = max_width
        target_height = int(target_width / aspect_ratio)

    fg_resized = fg_cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)

    padding_px = int(template_height * padding)
    if position == 'center':
        x = (template_width - target_width) // 2
        y = (template_height - target_height) // 2
    elif position == 'top-center':
        x = (template_width - target_width) // 2
        y = padding_px
    elif position == 'bottom-center':
        x = (template_width - target_width) // 2
        y = template_height - target_height - padding_px
    elif position == 'left':
        x = padding_px
        y = template_height - target_height - padding_px
    elif position == 'right':
        x = template_width - target_width - padding_px
        y = template_height - target_height - padding_px
    else:
        x = (template_width - target_width) // 2
        y = template_height - target_height - padding_px

    # Paste using alpha channel
    template.paste(fg_resized, (x, y), fg_resized)

    return template.convert('RGB')


# ================== INIT ==================

templates = get_available_templates()
print("Available templates:", templates if templates else "None (using default gradient)")


# ================== FLASK ROUTES ==================

@app.route('/')
def home():
    templates = get_available_templates()
    template_list = ''.join([f'<option value="{t}">{t}</option>' for t in templates])

    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Background Removal - rembg (No Custom Mask)</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; background: #1a1a2e; color: #eee; }}
            .upload-form {{ background: #16213e; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
            input[type="file"] {{ margin: 10px 0; color: #eee; }}
            select, input[type="number"] {{ padding: 8px; margin: 5px; border-radius: 4px; border: 1px solid #0f3460; background: #1a1a2e; color: #eee; }}
            button {{ background: #e94560; color: white; padding: 12px 24px; border: none; cursor: pointer; border-radius: 5px; font-size: 16px; }}
            button:hover {{ background: #ff6b6b; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; background: #0f3460; color: #4ecca3; }}
            code {{ background: #0f3460; padding: 2px 6px; border-radius: 3px; }}
            .form-group {{ margin: 15px 0; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; color: #4ecca3; }}
            h1, h2, h3 {{ color: #4ecca3; }}
            .badge {{ background: #4ecca3; color: #1a1a2e; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
            .feature {{ background: #0f3460; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🎨 Background Removal <span class="badge">REMBG - NO MASK</span></h1>
        <div class="status">✅ Using rembg library (handles everything internally)</div>

        <div class="feature">
            <h3>🚀 New Approach - Zero Custom Mask Code:</h3>
            <p>Uses the <strong>rembg</strong> library which handles all background removal internally. No custom U2-Net mask code, no edge detection, no erosion - just clean removal.</p>
            <p><strong>Installation:</strong> <code>pip install rembg</code></p>
        </div>

        <div class="upload-form">
            <h2>Upload Image</h2>
            <form action="/extract" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label>Image:</label>
                    <input type="file" name="image" accept="image/*" required>
                </div>

                <div class="form-group">
                    <label>Template:</label>
                    <select name="template">
                        <option value="">Random Template</option>
                        <option value="__default__">Default Gradient</option>
                        {template_list}
                    </select>
                </div>

                <div class="form-group">
                    <label>Position:</label>
                    <select name="position">
                        <option value="bottom-center">Bottom Center</option>
                        <option value="center">Center</option>
                        <option value="top-center">Top Center</option>
                        <option value="left">Left</option>
                        <option value="right">Right</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Scale (0.1-1.0):</label>
                    <input type="number" name="scale" value="0.8" min="0.1" max="1.0" step="0.1">
                </div>

                <button type="submit">🚀 Remove Background (rembg)</button>
            </form>
        </div>

        <h2>API Endpoints</h2>
        <ul>
            <li><code>POST /extract</code> – remove background + place on template</li>
            <li><code>POST /cutout</code> – return transparent PNG only</li>
            <li><code>GET /templates</code> – list available templates</li>
            <li><code>GET /health</code> – health check</li>
        </ul>

        <h3>Install rembg:</h3>
        <code style="display: block; padding: 10px; background: #0f3460; border-radius: 5px; margin: 10px 0;">
            pip install rembg
        </code>
    </body>
    </html>
    '''


@app.route('/templates', methods=['GET'])
def list_templates():
    templates = get_available_templates()
    return jsonify({
        'templates': templates,
        'count': len(templates),
        'folder': TEMPLATE_FOLDER
    })


@app.route('/templates/<template_name>', methods=['GET'])
def preview_template(template_name):
    template_path = get_template_path(template_name)
    if template_path and os.path.exists(template_path):
        return send_file(template_path, mimetype='image/png')
    return jsonify({'error': f'Template not found: {template_name}'}), 404


@app.route('/extract', methods=['POST'])
def extract():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    template_name = request.form.get('template', '')
    position = request.form.get('position', 'bottom-center')
    scale = request.form.get('scale', 0.8, type=float)
    scale = max(0.1, min(1.0, scale))

    try:
        image = Image.open(file.stream).convert('RGB')

        # Remove background using rembg (no custom mask)
        cutout_rgba = remove_background_rembg(image)

        # Choose template
        if template_name == '__default__':
            template_path = None
        elif template_name:
            template_path = get_template_path(template_name)
            if not template_path:
                return jsonify({'error': f'Template not found: {template_name}'}), 404
        else:
            template_path = get_template_path()

        # Place on template
        result = place_rgba_on_template(
            cutout_rgba,
            template_path=template_path,
            position=position,
            scale=scale
        )

        img_io = io.BytesIO()
        result.save(img_io, 'PNG', quality=95)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', as_attachment=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/cutout', methods=['POST'])
def cutout():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        image = Image.open(file.stream).convert('RGB')
        cutout_rgba = remove_background_rembg(image)

        img_io = io.BytesIO()
        cutout_rgba.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    templates = get_available_templates()
    return jsonify({
        'status': 'healthy',
        'method': 'rembg library (no custom mask)',
        'templates_available': len(templates),
        'templates': templates
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)