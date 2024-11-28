from flask import Flask, Response, render_template, request, redirect, url_for,jsonify
import cv2
from PIL import Image
import os
import numpy as np
import base64
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
import pickle
import logging
logging.basicConfig(level=logging.DEBUG)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        in_features = self.resnet.fc.in_features  
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.authorized_embeddings = {}
        try:
            with open(r'static/admin_user\admin_embeddings.pkl', 'rb') as f:
                self.admin_embeddings = pickle.load(f)

            with open(r'static/admin_user\admin_details.pkl', 'rb') as f:
                self.admin_details = pickle.load(f)
            print('load_old_suses')
        except FileNotFoundError:
            self.admin_embeddings = {}
            self.admin_details = {}


    def forward_one(self, x):
        embedding = self.resnet(x)  
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

    def autherise_user(self, image, name,admin_bool=False,admin_dict={}):
        self.eval()
        with torch.no_grad():
            embedding = self.forward_one(image)
        if admin_bool:
            self.admin_embeddings[name] = embedding
            self.admin_details[name] = admin_dict
            with open(r'static/admin_user\admin_embeddings.pkl', 'wb') as f:
                pickle.dump(self.admin_embeddings, f)

            with open(r'static/admin_user\admin_details.pkl', 'wb') as f:
                pickle.dump(self.admin_details, f)
        else:
            self.authorized_embeddings[name] = embedding
    
    def verify_autherisation(self, image, threshold=0.5,admin_bool=False):
        self.eval()
        with torch.no_grad():
            embedding_new = self.forward_one(image)

        closest_distance = float('inf')
        closest_user = None
        embeddings = self.authorized_embeddings
        if admin_bool:
            embeddings = self.admin_embeddings
        for user, embedding in embeddings.items():
            distance = torch.pairwise_distance(embedding_new, embedding).item()
            app.logger.debug(f"User: {user}, Distance: {distance}")
            if distance < closest_distance:
                closest_distance = distance
                closest_user = user
        if closest_distance < threshold:
            return closest_user, closest_distance
        else:
            return '', closest_distance

def transform_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    return transform(image).unsqueeze(0)



app = Flask(__name__)
camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
save_folder = 'authorized_user_images'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

if not os.path.exists('static/admin_user'):
    os.makedirs('static/admin_user')

for file_name in os.listdir(save_folder):
    file_path = os.path.join(save_folder, file_name)

    if os.path.isfile(file_path):
        os.remove(file_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork()
state_dict = torch.load("resnet50.pkl", weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

admin_user_temp = ""

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/authorize_user', methods=['POST'])
def authorize_user():
    name = request.form.get('name')  
    if not name:
        return jsonify({"error": "Name is required!"}), 400

    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image from camera."}), 500
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"error": "No face detected in the frame!"}), 400

    x, y, w, h = faces[0]
    cropped_face = frame[y:y+h, x:x+w]
    cropped_face = cv2.resize(cropped_face, (224, 224)) 

    authorized_image = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    authorized_image = transform_image(authorized_image)
    authorized_image = authorized_image.to(device)
    model.autherise_user(authorized_image,name)

    temp_image_path = os.path.join('authorized_user_images', f"{name}.jpg") 
    cv2.imwrite(temp_image_path, cropped_face)

    return jsonify({"message": "User authorized successfully!", "image_path": temp_image_path})

@app.route('/admin_user', methods=['POST'])
def admin_user():
    success, frame = camera.read()
    if not success:
        app.logger.error("Camera failed to capture an image.")
        return jsonify({"error": "Failed to capture image from camera."}), 500

    form_data = request.form.to_dict()
    name = form_data.get('name')
    if not form_data:
        return jsonify({"error": "data is required!"}), 400
    if not name:
        return jsonify({"error": "Name is required!"}), 400
    

    admin_dict = {'major':form_data.get('major'),'id_number':form_data.get('id_number'),
                  "email":form_data.get('email'),'address':form_data.get('address'),
                  "phone":form_data.get('phone'),'linkedin':form_data.get('linkedin'),'image_src':f'static/admin_user/{name}.jpg'}
   
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        app.logger.warning("No face detected in the captured frame.")
        return jsonify({"error": "No face detected in the frame!"}), 400

    x, y, w, h = faces[0]
    cropped_face = frame[y:y+h, x:x+w]
    cropped_face = cv2.resize(cropped_face, (224, 224)) 

    authorized_image = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    app.logger.debug(f"Verify image shape before transformation: {authorized_image.shape}")
    authorized_image = transform_image(authorized_image)
    authorized_image = authorized_image.to(device)
    model.autherise_user(authorized_image,name,True,admin_dict)

    temp_image_path = os.path.join('static/admin_user', f"{name}.jpg") 
    cv2.imwrite(temp_image_path, cropped_face)

    return jsonify({"message": "User authorized successfully!", "image_path": temp_image_path})

@app.route('/verify_user', methods=['POST'])
def verify_user():
    global admin_user_temp
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture image from camera."}), 500
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"error": "No face detected in the frame!"}), 400

    x, y, w, h = faces[0]
    cropped_face = frame[y:y+h, x:x+w]

    cropped_face = cv2.resize(cropped_face, (224, 224))  

    verify_image = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    verify_image = transform_image(verify_image)
    verify_image = verify_image.to(device)
    user = ''
    user, _ = model.verify_autherisation(verify_image,0.5)  
    admin_user = ''
    admin_user, _ = model.verify_autherisation(verify_image,0.5,True) 
      
    if admin_user:
        admin_user_temp = admin_user
        return jsonify({"message": f"Admin user: {admin_user}!"})
        
    if user:
        return jsonify({"message": f"Autherised user: {user}!"})

    return jsonify({"message": "Unautherised user!"})
 
@app.route('/post_verification')
def post_verification():
    global admin_user_temp
    name = admin_user_temp
    admin_user_temp = ''

    return render_template('admin_id.html', name=name, major= model.admin_details[name]['major'], id_number= model.admin_details[name]['id_number'], 
                           email=model.admin_details[name]['email'], address =model.admin_details[name]['address'], 
                           phone=model.admin_details[name]['phone'], linkedin =model.admin_details[name]['linkedin'], image_src = model.admin_details[name]['image_src'])


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
    'admin_user'