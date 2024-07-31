##main.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16, mobilenet_v2, efficientnet_b0
from src import dbconnect as db
from bson.objectid import ObjectId  # Import ObjectId
from PIL import Image
from datetime import datetime  # Import datetime
import base64
import io
import json

app = Flask(__name__)
CORS(app)

# Load pre-trained models with the new 'weights' parameter
resnet18_model = resnet18(weights='IMAGENET1K_V1').eval()
vgg16_model = vgg16(weights='IMAGENET1K_V1').eval()
mobilenet_v2_model = mobilenet_v2(weights='IMAGENET1K_V1').eval()
efficientnet_b0_model = efficientnet_b0(weights='IMAGENET1K_V1').eval()

models = {
    'resnet18': resnet18_model,
    'vgg16': vgg16_model,
    'mobilenet_v2': mobilenet_v2_model,
    'efficientnet_b0': efficientnet_b0_model
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_questionnaire_score(responses):
    score = responses.count('yes')
    return score

def get_image_prediction(model, img):
    img_t = transform(img)
    img_t = img_t.unsqueeze(0)
    with torch.no_grad():
        output = model(img_t)
    _, predicted = torch.max(output, 1)
    probability = torch.softmax(output, dim=1)[0, predicted].item()
    return probability

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = db.mongodb_find_one(data={"name": username, "password": password}, collection='ASD_logs', db='local')
    if user:
        return jsonify({"message": "Login successful", "user_id": str(user['_id'])}), 200
    else:
        return jsonify({"message": "Invalid username or password!!"}), 401



@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # Check if user already exists
    existing_user = db.mongodb_find_one(data={"name": username}, collection='ASD_logs', db='local')
    if existing_user:
        return jsonify({"message": "Username already exists"}), 409

    # Add new user to the database
    new_user = {"name": username, "password": password}
    db.mongodb_insert(new_user, collection='ASD_logs', db='local')
    return jsonify({"message": "Sign-up successful"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image'].read()
    name = request.form.get('name')
    age = request.form.get('age')
    responses = json.loads(request.form.get('responses'))
    selected_model = request.form.get('model')

    img = Image.open(io.BytesIO(img_file)).convert('RGB')

    if selected_model in models:
        model = models[selected_model]
        image_probability = get_image_prediction(model, img)
    else:
        return jsonify({"error": "Invalid model selected"}), 400

    questionnaire_score = calculate_questionnaire_score(responses)
    final_asd_probability = (0.7 * image_probability) + (0.3 * (questionnaire_score / len(responses)))

    return jsonify({
        "name": name,
        "age": age,
        "responses": responses,
        "selected_model": selected_model,
        "ASD_Probability": float(final_asd_probability)
    })


@app.route('/ASD_predicted_result', methods=['POST'])
def ASD_predicted_result():
    data = request.json
    name = data.get('name')
    age = data.get('age')
    image_base64 = data.get('image')
    ## show image size in python
    selected_model = data.get('selected_model')
    asd_probability = data.get('ASD_Probability')

    # Get the current date and time
    created_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Decode the base64 image
    image_data = base64.b64decode(image_base64)

    # Load the image using PIL
    image = Image.open(io.BytesIO(image_data))

    # Get the size of the image in bytes and convert to kilobytes
    image_size_bytes = len(image_data)
    image_size_kb = image_size_bytes / 1024

    # Print the image size in KB (or use it as needed)
    print(f"Image size: {image_size_kb:.2f} KB")

    # Validate the received data
    if not all([name, age, image_base64, selected_model, asd_probability is not None]):
        return jsonify({"error": "Missing data in the request"}), 400

    # Check if the record already exists based on name and image
    existing_record = db.mongodb_find_one(
        data={"name": name},
        collection='ASD_results',
        db='local'
    )

    if existing_record:
        # Update the existing record
        update_data = {
            "_id": ObjectId(existing_record['_id']),
            "name": name,
            "age": age,
            "image_base64": image_base64,
            "selected_model": selected_model,
            "ASD_Probability": asd_probability,
            "created_date": created_date
        }
        db.mongodb_delete_one(data={"_id": existing_record['_id']}, collection='ASD_results', db='local')
        db.mongodb_insert(update_data, collection='ASD_results', db='local')
        return jsonify({"message": "Prediction result updated successfully"}), 200
    else:
        # Insert a new record
        record = {
            "name": name,
            "age": age,
            "image_base64": image_base64,
            "selected_model": selected_model,
            "ASD_Probability": asd_probability,
            "created_date": created_date
        }
        db.mongodb_insert(record, collection='ASD_results', db='local')
        return jsonify({"message": "Prediction result stored successfully"}), 200


@app.route('/ASD_results', methods=['GET'])
def ASD_results():
    try:
        # Fetch all records from the ASD_results collection
        results = db.mongodb_find(data={}, collection='ASD_results', db='local')

        # Convert MongoDB records to a list of dictionaries
        results_list = []
        for result in results:
            result['_id'] = str(result['_id'])  # Convert ObjectId to string
            results_list.append(result)

        return jsonify(results_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
