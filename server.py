import uuid
from flask import Flask, request, jsonify, render_template
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import os
import json
import sqlite3
from werkzeug.utils import secure_filename

app = Flask(__name__)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_embedding_from_upload(file):
    img = Image.open(file)
    img_cropped, prob = mtcnn(img, return_prob=True)
    if img_cropped is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        img_aligned = img_cropped.unsqueeze(0).to(device)
        embedding = resnet(img_aligned).detach().cpu().squeeze(0)
        return embedding
    else:
        print('No face detected in the image.')
        return None



@app.route('/add_profile_page')
def add_profile_page():
    return render_template('add.html')

# Route to render the search profiles page
@app.route('/')
def search_profiles_page():
    return render_template('index.html')



# Route to add a profile
@app.route('/add_profile', methods=['POST'])
def add_profile():
    data = request.form
    ID = str(uuid.uuid4())
    name = data['name']
    gender = data['gender']  
    age = data['age']
    height = data['height']
    place = data['place']
    health_status = data['cas']

    print(request.files)

    # Check if the file is present in the request
    if 'file' not in request.files:
        return 'Error: No file part'

    file = request.files['file']

    if file.filename == '':
        return 'Error: No selected file'

    if not (file and allowed_file(file.filename)):
        return 'Error: Invalid file extension'
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Save the embedding to a file
    embedding = get_embedding_from_upload(file_path)
    if embedding is None:
        return 'Error: No face detected in the image.'
    
    embedding_path = os.path.join('embeddings', '{}.em'.format(ID))
    torch.save(embedding, embedding_path)
    print('Embedding saved to: {}'.format(embedding_path))

    # Save profile data to SQLite database as JSON string
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO profiles (id, image,  data) VALUES (?, ?, ?)",
                    (ID, json.dumps({
                                'name': name,
                                'gender': gender,
                                'age': age,
                                'height': height,
                                'place': place,
                                'health_status': health_status
                    }), file_path))
    conn.commit()
    conn.close()

    return render_template('index.html')
            
    

# Route to search for the top 3 closest profiles
@app.route('/search_profiles', methods=['POST'])
def search_profiles():
    if 'file' not in request.files:
        return 'Error: No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'Error: No selected file'

    if not (file and allowed_file(file.filename)):
        return 'Error: Invalid file extension'

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    input_embedding = get_embedding_from_upload(file_path)
    if input_embedding is None:
        return 'Error: No face detected in the image.'

    embeddings_folder = 'embeddings'
    scores = []

    for file_name in os.listdir(embeddings_folder):
        if file_name.endswith('.em'):
            file_path = os.path.join(embeddings_folder, file_name)
            saved_embedding = torch.load(file_path)
            closeness_score = (input_embedding - saved_embedding).norm().item()
            scores.append((file_name[:-3], closeness_score))

    # Sort by closeness score
    scores.sort(key=lambda x: x[1])
    # Return the top 3 IDs and scores along with profile data
    top_3_ids = [score for score in scores[:4]]
    top_3_data = []

    # Retrieve profile data from the database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    for profile_id, score in top_3_ids:
        cursor.execute("SELECT id, image, data FROM profiles WHERE id=?", (profile_id,))
        data_json = cursor.fetchone()
        if data_json:
            id, data, image = data_json
            top_3_data.append({'id': profile_id, 'image': image, 'data': json.loads(data), 'score': score})
    conn.close()

    return render_template('find.html', search_results=top_3_data)
        

if __name__ == '__main__':
    app.run(debug=True)
