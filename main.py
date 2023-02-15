from flask import Flask, jsonify

from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition
import os 
import numpy as np

app = Flask(__name__)

# Load known faces and names
known_faces = []
known_names = []

training_images_dir = "Training_images"
training_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images/Training_images')

for filename in os.listdir(training_images_dir):
    # Load image
    image = face_recognition.load_image_file(f"{training_images_dir}/{filename}")
    # Get face encoding
    face_encoding = face_recognition.face_encodings(image)[0]
    # Add to known faces and names
    known_faces.append(face_encoding)
    known_names.append(filename.split(".")[0])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")

@app.route("/signup", methods=["POST"])
def signup_post():
    # Get user name from form
    user_name = request.form.get("username")
    if user_name is None or user_name == "":
        return "Username is required"

    # Start webcam
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    # Convert frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Get face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Save image file
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        image_path = f"{training_images_dir}/{user_name}.jpg"
        cv2.imwrite(image_path, frame)
        known_faces.append(face_encoding)
        known_names.append(user_name)
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('index'))
    else:
        cap.release()
        cv2.destroyAllWindows()
        return "No face detected, please try again"

@app.route("/login", methods=["POST"])
def login():
    # Start webcam
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    # Convert frame to RGB
    rgb_frame = frame[:, :, ::-1]

    # Get face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Check if face encoding matches with known face encoding
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            username = known_names[best_match_index]
            cap.release()
            cv2.destroyAllWindows()
            return redirect(url_for('home', username=username))
    cap.release()
    cv2.destroyAllWindows()
    return "Face not recognized, please try again"


@app.route("/home/<username>")
def home(username):
    return render_template("home.html", username=username)

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
