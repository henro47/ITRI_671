from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return render(request, "home.html")

def prediction(request):
    return render(request, "prediction.html")