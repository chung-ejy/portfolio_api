import warnings
warnings.filterwarnings(action='ignore')
import pickle
class Modeler(object):

    @classmethod
    def predict(self,models,prediction_set,factors):
        factors = [x for x in factors if x != "y" and x != "ticker"]
        for row in models.iterrows():
            try:
                api = row[1]["api"]
                if api == "tf":
                    continue
                else:
                    model = pickle.loads(row[1]["model"])
                    score = row[1]["score"]
                    prediction_set[f"{api}_prediction"] = model.predict(prediction_set[factors])
                prediction_set[f"{api}_score"] = score 
            except Exception as e:
                print(str(e))
        return prediction_set

   