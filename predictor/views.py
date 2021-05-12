from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse
from rest_framework.views import APIView
import numpy as np
import datetime
# Create your views here.

class call_model(APIView):
    def get(self,request):
        crops_dict={'Rice': 'https://cdn.britannica.com/17/176517-050-6F2B774A/Pile-uncooked-rice-grains-Oryza-sativa.jpg', 'Maize': 'https://www.mccormick.it/wp-content/uploads/2020/05/dati-aggiornati-sulla-produzione-di-mais-nel-mondo-853x480.jpg', 'Chickpea': 'https://www.inspiredtaste.net/wp-content/uploads/2016/06/How-to-Cook-Chickpeas-Recipe-2-1200.jpg', 'Kidneybeans': 'https://vinayakfoodsgroup.com/cjh/pulses-grains/pulses-grains-img/red-kidney-beans-concept-mob.png', 'Pigeonpeas': 'https://i.ndtvimg.com/mt/cooks/2014-11/yellow-lentils-arhar-dal.jpg', 'Mothbeans': 'https://www.ugaoo.com/media/wysiwyg/pigeonpea_hwh1.jpg', 'Mungbean': 'https://media.istockphoto.com/photos/mung-beans-isolated-on-white-picture-id1061909094?k=6&m=1061909094&s=612x612&w=0&h=sqNvTSU5PXuPN-QvHWoCX9QqBftYY1-Lwy5H7_6_tAQ=', 'Blackgram': 'https://upload.wikimedia.org/wikipedia/commons/6/6f/Black_gram.jpg', 'Lentil': 'https://upload.wikimedia.org/wikipedia/commons/6/6f/Black_gram.jpg', 'Pomegranate': 'https://static.toiimg.com/thumb/msid-69940581,width-800,height-600,resizemode-75,imgsize-1620068,pt-32,y_pad-40/69940581.jpg', 'Banana': 'https://cdn.mos.cms.futurecdn.net/42E9as7NaTaAi4A6JcuFwG-1200-80.jpg', 'Mango': 'https://i0.wp.com/cdn-prod.medicalnewstoday.com/content/images/articles/322/322096/mangoes-chopped-and-fresh.jpg?w=1155&h=1541', 'Grapes': 'https://specialtyproduce.com/sppics/1224.png', 'Watermelon': 'https://snaped.fns.usda.gov/sites/default/files/styles/crop_ratio_7_5/public/seasonal-produce/2018-05/watermelon.jpg?itok=6EdNOdUo', 'Muskmelon': 'https://5.imimg.com/data5/KL/IA/MY-36648926/muskmelon-500x500.jpg', 'Apple': 'https://i2.wp.com/ceklog.kindel.com/wp-content/uploads/2013/02/firefox_2018-07-10_07-50-11.png?fit=641%2C618&ssl=1', 'Orange': 'https://media.istockphoto.com/photos/orange-picture-id185284489?k=6&m=185284489&s=612x612&w=0&h=x_w4oMnanMTQ5KtSNjSNDdiVaSrlxM4om-3PQTIzFaY=', 'Papaya': 'https://rukminim1.flixcart.com/image/416/416/j7td5e80/plant-seed/h/x/v/10-red-lady-taiwan-papaya-seeds-green-world-original-imaeqnhyztffgbah.jpeg?q=70', 'Coconut': 'https://images.indianexpress.com/2020/11/coconut-tree-pixabay-1200.jpg', 'Cotton': 'https://images.ctfassets.net/3s5io6mnxfqz/4TV7YTCO1DJuMhhn7RD1Ol/b5a6c12340e6529a86bc1b557ed2d8f8/AdobeStock_136921602.jpeg', 'Jute': 'https://5.imimg.com/data5/RT/SO/MY-25065022/jute-fiber-500x500.jpg', 'Coffee': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Roasted_coffee_beans.jpg/1200px-Roasted_coffee_beans.jpg'}
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
            crop=prediction.item().title()

            d=datetime.datetime.now()
            date=d.strftime("%B %d, %Y")

            # vectorize sound 
            response = {'crop': crop,"crop_image":crops_dict[crop],"date":date,"n":npk[0],"p":npk[1],"k":npk[2],"ph":ph,"temperature": temperature,"humidity":humidity,"rainfall":rainfall}          
            return JsonResponse(response)