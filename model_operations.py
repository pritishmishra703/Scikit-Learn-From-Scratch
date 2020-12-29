import pickle
import os
import requests
from bs4 import BeautifulSoup
import json

class save_load_model():
    def save(self, model,file_name, path=""):
        saved_model = pickle.dumps(model)
        fulldir = os.path.join(path, file_name)
        with open(fulldir, "bw") as f:
            f.write(saved_model)

    def load_model(self, path):
        with open(path, "br") as f:
            readed_model = f.read()
            loaded_model = pickle.loads(readed_model)

        return loaded_model

class gpfc:
    def get_predictions(self, api_key, data):
        self.api_key = api_key
        request_link = f"http://127.0.0.1:8000/api-key/?token={self.api_key}&X={data}"
        pred_request = requests.get(request_link)
        predictions = BeautifulSoup(pred_request.content, "html.parser")
        print(str(predictions))
        predictions = json.loads(str(predictions))
        return predictions
