from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datacruncher.datacruncher import Datacruncher
from database.adatabase import ADatabase
# Create your views here.

@csrf_exempt
def apiView(request):
    try:
        if request.method == "GET":
            complete = Datacruncher.blog_cruncher()
        elif request.method == "POST":
            data = request.body
            complete = Datacruncher.factory(data)
        else:
            complete = {}
    except Exception as e:
        complete = {"data":[],"errors":str(e)}
    return JsonResponse(complete,safe=False)

@csrf_exempt
def visualizationView(request):
    try:
        if request.method == "GET":
            strategy_name = request.GET.get('strategy_name', None)  # Get the strategy_name from the GET parameters
            db = ADatabase(strategy_name)
            db.cloud_connect()
            data = db.retrieve("visualization").fillna(0).to_dict("records")
            db.disconnect()
            complete = {"visualization":data}
        else:
            complete = {}
    except Exception as e:
        complete = {"data":[],"errors":str(e)}
    return JsonResponse(complete,safe=False)

@csrf_exempt
def tradesView(request):
    try:
        if request.method == "GET":
            strategy_name = request.GET.get('strategy_name', None)  # Get the strategy_name from the GET parameters
            db = ADatabase(strategy_name)
            db.cloud_connect()
            data = db.retrieve("trades").fillna(0).to_dict("records")
            db.disconnect()
            complete = {"trades":data}
        else:
            complete = {}
    except Exception as e:
        complete = {"data":[],"errors":str(e)}
    return JsonResponse(complete,safe=False)

@csrf_exempt
def rafView(request):
    try:
        if request.method == "GET":
            db = ADatabase("raf")
            db.cloud_connect()
            data = db.retrieve("screener").fillna(0).to_dict("records")
            db.disconnect()
            complete = {"screener":data}
        else:
            complete = {}
    except Exception as e:
        complete = {"data":[],"errors":str(e)}
    return JsonResponse(complete,safe=False)