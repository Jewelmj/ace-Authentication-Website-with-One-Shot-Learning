from flask import Flask, Response, render_template, request,jsonify
import pyodbc
import cv2
from PIL import Image
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from mtcnn import MTCNN
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

    def forward_one(self, x):
        embedding = self.resnet(x)  
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    
class Load_sql():
    def __init__(self):
        connection_string = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=Jewel;DATABASE=User_Face_verification_system;Trusted_Connection=yes;'
        self.connection = pyodbc.connect(connection_string)
        self.cursor = self.connection.cursor()

        try:
            with open(r'static/visiter_user\visiter_embeddings.pkl', 'rb') as f:
                self.visiter_embeddings = pickle.load(f)
        except FileNotFoundError:
            self.visiter_embeddings = {}
        try:
            with open(r'static/authorized_user\authorized_embeddings.pkl', 'rb') as f:
                self.authorized_embeddings = pickle.load(f)
        except FileNotFoundError:
            self.authorized_embeddings = {}
        try:
            with open(r'static/admin_user\admin_embeddings.pkl', 'rb') as f:
                self.admin_embeddings = pickle.load(f)
        except FileNotFoundError:
            self.admin_embeddings = {}
        try:
            temp_list = ['major' , 'id_number', 'email', 'address', 'phone','linkedin','image_src']
            self.cursor.execute('SELECT * FROM UserAdmin')
            rows = self.cursor.fetchall() 
            self.admin_details = {} 
            for i in range(len(rows)):
                temp_dict = {}
                for j in range(len(rows[i][2:])):
                    temp_dict[temp_list[j]] = rows[i][2:][j]
                self.admin_details[rows[i][1]] = temp_dict
        except FileNotFoundError:
            self.admin_details = {}
        try:
            temp_list = ['major' , 'id_number', 'email', 'address', 'phone','linkedin','image_src']
            self.cursor.execute('SELECT * FROM UserStudent')
            rows = self.cursor.fetchall() 
            self.authorized_details = {} 
            for i in range(len(rows)):
                temp_dict = {}
                for j in range(len(rows[i][2:])):
                    temp_dict[temp_list[j]] = rows[i][2:][j]
                self.authorized_details[rows[i][1]] = temp_dict
        except FileNotFoundError:
            self.authorized_details = {}
        try:
            temp_list = ['major' , 'id_number', 'email', 'address', 'phone','linkedin','image_src']
            self.cursor.execute('SELECT * FROM UserVisiter')
            rows = self.cursor.fetchall() 
            self.visiter_details = {} 
            for i in range(len(rows)):
                temp_dict = {}
                for j in range(len(rows[i][2:])):
                    temp_dict[temp_list[j]] = rows[i][2:][j]
                self.visiter_details[rows[i][1]] = temp_dict
        except FileNotFoundError:
            self.visiter_details = {}

    def autherise_user(self, image, name,user=0,admin_dict={}): # 0: student, 1: admin, 2:visitor
        model.eval()
        with torch.no_grad():
            embedding = model.forward_one(image)
        if user == 1:
            self.admin_embeddings[name] = embedding
            self.admin_details[name] = admin_dict
            with open(r'static/admin_user\admin_embeddings.pkl', 'wb') as f:
                pickle.dump(self.admin_embeddings, f)
            
            sql_insert = '''
                INSERT INTO UserAdmin (Name, Major, IDNumber, Email, Address, Phone, LinkedIn,IMG_SCR)
                VALUES (?, ?, ?, ?, ?, ?, ?,?)
                '''
            temp_list = [name]
            for val in admin_dict.values():
                temp_list.append(val)
            values = temp_list
            self.cursor.execute(sql_insert, values)
            self.connection.commit()
        elif user == 2:
            self.visiter_embeddings[name] = embedding
            self.visiter_details[name] = admin_dict
            with open(r'static/visiter_user\visiter_embeddings.pkl', 'wb') as f:
                pickle.dump(self.visiter_embeddings, f)

            sql_insert = '''
                INSERT INTO UserVisiter (Name, Major, IDNumber, Email, Address, Phone, LinkedIn,IMG_SCR)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
            temp_list = [name]
            for val in admin_dict.values():
                temp_list.append(val)
            values = temp_list
            self.cursor.execute(sql_insert, values)
            self.connection.commit()
        else:
            self.authorized_embeddings[name] = embedding
            self.authorized_details[name] = admin_dict
            with open(r'static/authorized_user\authorized_embeddings.pkl', 'wb') as f:
                pickle.dump(self.authorized_embeddings, f)

            sql_insert = '''
                INSERT INTO UserStudent (Name, Major, IDNumber, Email, Address, Phone, LinkedIn,IMG_SCR)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                '''
            temp_list = [name]
            for val in admin_dict.values():
                temp_list.append(val)
            values = temp_list
            self.cursor.execute(sql_insert, values)
            self.connection.commit()

    def verify_autherisation(self, image, threshold=0.3,user=0):
        model.eval()
        with torch.no_grad():
            embedding_new = model.forward_one(image)

        closest_distance = float('inf')
        closest_user = None
        embeddings = self.authorized_embeddings
        if user==1:
            embeddings = self.admin_embeddings
        elif user==2:
            embeddings = self.visiter_embeddings
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

def crop_img():
    success, frame = camera.read()
    if not success:
        app.logger.error("Camera failed to capture an image.")
        return jsonify({"error": "Failed to capture image from camera."}), 500
    
    faces = detector.detect_faces(frame)
    if len(faces) == 0:
        app.logger.warning("No face detected in the captured frame.")
        return jsonify({"error": "No face detected in the frame!"}), 400

    face = faces[0]
    if 'box' in face:  
        x, y, w, h = face['box']  
        cropped_face = frame[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (224, 224))
        return cropped_face
    else:
        app.logger.warning("Face detection did not return the expected bounding box format.")
        return jsonify({"error": "Face detection failed to return a valid bounding box."}), 500

def delete_all_files_in_folder(folder_path):
    try:
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' does not exist.")
            return
        
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        
        print(f"All contents of '{folder_path}' have been deleted.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

app = Flask(__name__)
camera = cv2.VideoCapture(0)
detector = MTCNN()

if not os.path.exists('static/admin_user'):
    os.makedirs('static/admin_user')

if not os.path.exists('static/authorized_user'):
    os.makedirs('static/authorized_user')

if not os.path.exists('static/visiter_user'):
    os.makedirs('static/visiter_user')

load = Load_sql()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork()
state_dict = torch.load("resnet50.pkl", weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

admin_user_temp = ""
user_type_temp = 0

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        faces = detector.detect_faces(frame)
        
        for face in faces:
            if 'box' in face:
                x, y, w, h = face['box']
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

@app.route('/add_user')
def add_user():
    cropped_face = crop_img()
    mirrored_face = cv2.flip(cropped_face, 1)

    try:
        cv2.imwrite('static/temp_img/test.jpg', mirrored_face)
    except:
        file_path = 'static/temp_img/test.jpg'
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted.")
        else:
            print(f"File {file_path} does not exist.")
        return jsonify({"message": "No face!"})

    return jsonify({"message": "photo taken!"})

@app.route('/add_user_verify',methods=['GET'])
def add_user_verify():
    return render_template('add_user.html')

@app.route('/authorize_user', methods=['POST'])
def authorize_user():
    form_data = request.form.to_dict()
    name = form_data.get('name')
    if not form_data:
        return jsonify({"error": "data is required!"}), 400
    if not name:
        return jsonify({"error": "Name is required!"}), 400
    

    autherized_dict = {'major':form_data.get('major'),'id_number':form_data.get('id_number'),
                  "email":form_data.get('email'),'address':form_data.get('address'),
                  "phone":form_data.get('phone'),'linkedin':form_data.get('linkedin'),'image_src':f'static/authorized_user/{name}.jpg'}
   
    cropped_face =  cv2.imread('static/temp_img/test.jpg')

    authorized_image = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    app.logger.debug(f"Verify image shape before transformation: {authorized_image.shape}")
    authorized_image = transform_image(authorized_image)
    authorized_image = authorized_image.to(device)
    load.autherise_user(authorized_image,name,0,autherized_dict)

    temp_image_path = os.path.join('static/authorized_user', f"{name}.jpg") 
    cv2.imwrite(temp_image_path, cropped_face)

    return jsonify({"message": "User authorized successfully!", "image_path": temp_image_path})

@app.route('/admin_user', methods=['POST'])
def admin_user():
    form_data = request.form.to_dict()
    name = form_data.get('name')
    if not form_data:
        return jsonify({"error": "data is required!"}), 400
    if not name:
        return jsonify({"error": "Name is required!"}), 400
    

    admin_dict = {'major':form_data.get('major'),'id_number':form_data.get('id_number'),
                  "email":form_data.get('email'),'address':form_data.get('address'),
                  "phone":form_data.get('phone'),'linkedin':form_data.get('linkedin'),'image_src':f'static/admin_user/{name}.jpg'}
   
    cropped_face = cv2.imread('static/temp_img/test.jpg')

    authorized_image = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    app.logger.debug(f"Verify image shape before transformation: {authorized_image.shape}")
    authorized_image = transform_image(authorized_image)
    authorized_image = authorized_image.to(device)
    load.autherise_user(authorized_image,name,1,admin_dict)

    temp_image_path = os.path.join('static/admin_user', f"{name}.jpg") 
    cv2.imwrite(temp_image_path, cropped_face)

    return jsonify({"message": "User authorized successfully!", "image_path": temp_image_path})

@app.route('/visiter_user', methods=['POST'])
def visiter_user():
    form_data = request.form.to_dict()
    name = form_data.get('name')
    if not form_data:
        return jsonify({"error": "data is required!"}), 400
    if not name:
        return jsonify({"error": "Name is required!"}), 400
    

    visiter_dict = {'major':form_data.get('major'),'id_number':form_data.get('id_number'),
                  "email":form_data.get('email'),'address':form_data.get('address'),
                  "phone":form_data.get('phone'),'linkedin':form_data.get('linkedin'),'image_src':f'static/admin_user/{name}.jpg'}
    
    cropped_face = cv2.imread('static/temp_img/test.jpg')

    app.logger.debug(f"Verify image shape before transformation: {authorized_image.shape}")
    authorized_image = transform_image(authorized_image)
    authorized_image = authorized_image.to(device)
    load.autherise_user(authorized_image,name,2,visiter_dict)

    temp_image_path = os.path.join('static/visiter_user', f"{name}.jpg") 
    cv2.imwrite(temp_image_path, cropped_face)

    return jsonify({"message": "User authorized successfully!", "image_path": temp_image_path})

@app.route('/verify_user', methods=['POST'])
def verify_user():
    cropped_face = crop_img()
    try:
        cv2.imwrite('static/temp_img/test.jpg', cropped_face)
    except:
        file_path = 'static/temp_img/test.jpg'
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted.")
        else:
            print(f"File {file_path} does not exist.")
        return jsonify({"message": "No face!"})
    return jsonify({"message": "verify photo taken!"})

@app.route('/verify_user_post',methods=['GET'])
def verify_user_post():
    return render_template('verify_user.html')

@app.route('/verify_user_post2', methods=['POST'])
def verify_user_post2():
    global admin_user_temp, user_type_temp
    cropped_face = cv2.imread('static/temp_img/test.jpg')

    verify_image = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    verify_image = transform_image(verify_image)
    verify_image = verify_image.to(device)
    user = ''
    user, _ = load.verify_autherisation(verify_image,0.5,0)  
    admin_user = ''
    admin_user, _ = load.verify_autherisation(verify_image,0.5,1) 
    visiter_user = ''
    visiter_user, _ = load.verify_autherisation(verify_image,0.5,2) 
      
    if admin_user:
        admin_user_temp = admin_user
        user_type_temp = 1
        return jsonify({"message": f"Admin user: {admin_user_temp}!"},{'data':{"admin_user_temp":admin_user_temp,"user_type_temp":user_type_temp}})
        
    if user:
        admin_user_temp = user
        user_type_temp = 0
        if user_type_temp == 0:
            details = load.authorized_details
        elif user_type_temp == 1:
            details = load.admin_details
        elif user_type_temp == 2:
            details = load.visiter_details
        return jsonify({"message": f"Autherised user: {admin_user_temp}!"},{'data':{"admin_user_temp":admin_user_temp,"user_type_temp":user_type_temp,'details':details}})
    
    if visiter_user:
        admin_user_temp = visiter_user
        user_type_temp = 2
        return jsonify({"message": f"Visitor user: {admin_user_temp}!"},{'data':{"admin_user_temp":admin_user_temp,"user_type_temp":user_type_temp}})

    return jsonify({"message": "Unautherised user!"},{'data':{"admin_user_temp":admin_user_temp,"user_type_temp":user_type_temp}})
 
@app.route('/post_verification', methods=['GET'])  #POST
def post_verification():
    # data = request.get_json()
    # name = data.get('admin_user_temp')
    # user_type= data.get('user_type_temp')
    name = request.args.get('admin_user_temp')
    user_type= request.args.get('user_type_temp')
    print(name,user_type)

    # if user_type == 0:
    #     details = load.authorized_details
    # elif user_type == 1:
    #     details = load.admin_details
    # elif user_type == 2:
    #     details = load.visiter_details
    # print(details)

    if user_type == '0':
        details = load.authorized_details
    elif user_type == '1':
        details = load.admin_details
    elif user_type == '2':
        details = load.visiter_details
    return render_template('admin_id.html', name=name, major= details[name]['major'], id_number= details[name]['id_number'], 
                        email=details[name]['email'], address =details[name]['address'], 
                        phone=details[name]['phone'], linkedin =details[name]['linkedin'], image_src = details[name]['image_src'])

@app.route('/reset',methods = ['POST'])
def reset1():
    delete_all_files_in_folder('static/admin_user')
    delete_all_files_in_folder('static/authorized_user')
    delete_all_files_in_folder('static/visiter_user')

    load.visiter_embeddings = {}
    load.authorized_embeddings = {}
    load.admin_embeddings = {}
    load.admin_details = {}
    load.authorized_details = {}
    load.visiter_details = {}

    sql_command = f"DELETE FROM UserAdmin"
    load.cursor.execute(sql_command)
    sql_command = f"DELETE FROM UserStudent"
    load.cursor.execute(sql_command)
    sql_command = f"DELETE FROM UserVisiter"
    load.cursor.execute(sql_command)
    load.connection.commit()
    return jsonify({"message": "Done reset!"})
 


if __name__ == "__main__":
    app.run(debug=True, threaded=True)