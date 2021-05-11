from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse
from rest_framework.views import APIView
import numpy as np
# Create your views here.

class call_model(APIView):
    def get(self,request):
        if request.method == 'GET':  
            temperature = float(request.GET.get('temperature'))
            humidity = float(request.GET.get('humidity'))
            ph = float(request.GET.get('ph'))
            rainfall = float(request.GET.get('rainfall'))

            npk_sample=[temperature,humidity,ph,rainfall]
            npk_arr=np.array(npk_sample).reshape(1,-1)
            pred=PredictorConfig.npkmodel.predict(npk_arr)

            npk=pred.reshape(-1).tolist()

            sample=npk+[temperature, humidity, ph, rainfall]
            single_sample=np.array(sample).reshape(1,-1)
            prediction=PredictorConfig.omodel.predict(single_sample)

            # vectorize sound 
            response = {'crop': prediction.item().title()}          
            return JsonResponse(response)