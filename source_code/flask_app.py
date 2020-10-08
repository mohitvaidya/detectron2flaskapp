from pull_blob import pull_main
from ffmpeg_main import pre_process
from odmainutil_batch import object_d2
from odmainutil_batch import load_model

import os
import requests
from flask import Flask,request
from flask_caching import Cache


app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'

cache = Cache()
cache.init_app(app)

@cache.cached(timeout = 10e8)
def model():
    return load_model()

@app.route("/", methods=['GET','POST'])
def app_main():

    os.system('rm /app/*mp4')
    message = request.get_json(force=True)
    video_id = message["ID"]
    fps  = int(message['FPS'])
    duration = int(message["duration"])
    lang = message['lang'].lower()
    container_client = message['container'].lower()
    
    pull_main(video_id=video_id, lang= lang, container_client = container_client)
    pre_process(video_id=video_id, fps=fps, trim_duration=duration)
    data = object_d2(video_id= f'{video_id}_', model= model())
    return data


if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

