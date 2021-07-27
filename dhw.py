from flask import Flask, jsonify, request
from classifier125hw import predictletter



app = Flask(__name__)

@app.route('/predictletter', methods=['POST'])

def predictnumber():
    image= request.files.get('digit')
    prediction = predictletter(image)
    return jsonify({'prediction': prediction}), 200

if __name__ == '__main__':
    app.run(debug=True)