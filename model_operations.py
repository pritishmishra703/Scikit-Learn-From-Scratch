import pickle
import os

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
