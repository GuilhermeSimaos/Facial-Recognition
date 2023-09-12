import base64
import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FOURCC, 0x32595559)
camera.set(cv2.CAP_PROP_FPS, 60)

# known_face_encodings = []
# known_face_names = []

desired_width = 640
desired_heigth = 480


def initialize_known_faces():
    image_folder = "images"
    for filename in os.listdir(image_folder):
        face_name = os.path.splitext(filename)[0]
        image_path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        if len(face_encoding) > 0:
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(face_name)
    return known_face_encodings, known_face_names


def generate_frames():
    process_this_frame = True
    while True:
        # Read camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            # Grab a single frame of video
            ret, frame = camera.read()

            # Only process every other frame of video to save time
            if process_this_frame:

                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/take_picture')
def take_picture():
    success, frame = camera.read()
    if success:
        frame = cv2.resize(frame, (desired_width, desired_heigth))

        ret, buffer = cv2.imencode('.jpeg', frame)
        frame = buffer.tobytes()
        return frame, 200, {'Content-Type': 'image/jpeg'}
    return jsonify({'error': 'Failed to capture image'})


@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        data = request.get_json()
        image_data = data['image_data']
        face_name = data['face_name']

        # Save the image with the nomeRosto as the filename
        image_filename = f"images/{face_name}.jpg"

        # Decode base64 image data and save it on the server
        image_data = image_data.replace('data:image/jpeg;base64,', '')  # Remove data URI prefix
        image_bytes = base64.b64decode(image_data)
        with open(image_filename, 'wb') as image_file:
            image_file.write(image_bytes)

        # Optionally, you can update your known_face_encodings and known_face_names arrays here.
        new_face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(image_filename))

        if len(new_face_encodings) == 0:
            return jsonify({'message': 'Nenhuma rosto encontrado na imagem'})
        elif len(new_face_encodings) > 1:
            return jsonify({'message': 'Mais de rosto encontrado na image'})
        else:
            known_face_encodings.append(new_face_encodings[0])
            known_face_names.append(face_name)

        return jsonify({'message': 'Rosto registrado com sucesso!'})
    except Exception as e:
        print(str(e))
        return jsonify({'message': 'Erro ao registrar o rosto.'}), 500


if __name__ == '__main__':
    known_face_encodings, known_face_names = initialize_known_faces()
    app.run(debug=True, host='0.0.0.0')
