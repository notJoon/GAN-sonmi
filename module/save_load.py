import pickle as pkl

class SaveLoadData:
    def __init__(self, model, filename: str) -> str:
        self.filname = 'saved_model.pkl'

    def save_model(self, model):
        pkl.dump(model, open(self.filename, 'wb'))
        return f"{model} has saved"
    
    def load_model(self, model, filename: str) -> None:
        pkl.load(open(model, 'rb'))