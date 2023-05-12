from modeler.modeler import Modeler as m

class UniversalModeler(object):

    def __init__(self):
        self.name = "universal"
   
    def recommend(self,models,data,factors):
        prediction_set = m.predict(models,data,factors)
        return prediction_set