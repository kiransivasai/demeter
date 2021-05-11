from django.apps import AppConfig
from django.conf import settings
import os
import pickle


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    path=os.path.join(settings.MODELS,"npkmodel.pkl")
    path2=os.path.join(settings.MODELS,"model.pkl")

    with open(path,"rb") as pickled:
        npkmodel=pickle.load(pickled)
    with open(path2,"rb") as pickled2:
        omodel=pickle.load(pickled2)
