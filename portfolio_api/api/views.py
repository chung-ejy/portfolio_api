from django.http.response import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datacruncher.datacruncher import Datacruncher
# Create your views here.

@csrf_exempt
def apiView(request):
    try:
        if request.method == "GET":
            complete = {}
        elif request.method == "DELETE":
            complete = {}
        elif request.method == "UPDATE":
            complete = {}
        elif request.method == "POST":
            data = request.body
            complete = Datacruncher.factory(data)
        else:
            complete = {}
    except Exception as e:
        complete = {"data":[],"errors":str(e)}
    return JsonResponse(complete,safe=False)