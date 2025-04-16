from . import views
from django.urls import path

urlpatterns = [
    path("",views.apiView,name="api"),
    path("/visualization",views.visualizationView,name="visualization"),
    path("/trades",views.tradesView,name="trades"),
    path("/raf",views.rafView,name="raf"),
    path("/raffifty",views.raffiftyView,name="raffifty")
]