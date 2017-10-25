from flask import Flask, request, send_file
import json
import os
import urllib
import tempfile
import subprocess
import argparse


parser = argparse.ArgumentParser(description='Start masking service')
parser.add_argument('--model', dest='model', help='path to trained model')

args = parser.parse_args()


app = Flask(__name__)


@app.route('/mask', methods=['POST'])
def mask():
    image_url = json.loads(request.get_data())['image']
    filename, _ = urllib.urlretrieve(image_url)
    output = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    output.close()
    subprocess.call(['th', 'createMask.lua', args.model, '-img', filename,
                     '-output', output.name])
    response = send_file(output.name, mimetype='image/jpeg')
    os.remove(filename)
    os.remove(output.name)
    return response


app.run(debug=False, host='0.0.0.0', port=8888)
