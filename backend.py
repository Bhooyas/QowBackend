from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet import preprocess_input
import tensorflow as tf
import base64
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
api_v1_cors_config = {
  "origins": ["*"]
}
CORS(app, resources={"/*": api_v1_cors_config})

interpreter = tf.lite.Interpreter(model_path="converted_quant_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route("/", methods=["POST"])
def home():
	imgstring = request.json["image"].replace("data:image/png;base64,","")
	img = np.array(Image.open(io.BytesIO(base64.b64decode(imgstring))).convert('RGB'), dtype=np.float32)
	# print(img.shape)
	interpreter.set_tensor(input_details[0]["index"], img.reshape(-1, 224, 224, 3))
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])[0][0]
	if output_data < 0.5:
		predicted = "healthycows"
		probability = (1 - output_data) * 100
	else:
		predicted = "lumpycows"
		probability = output_data * 100
	d = {"predicted": predicted, "probability": probability}
	return jsonify(d)

if __name__ == "__main__":
	# app.run()
	app.run(host='0.0.0.0')