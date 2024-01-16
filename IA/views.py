import csv

import numpy as np
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, DetailView
from django.core.files.images import ImageFile

from IA.forms import DataForm, OLSForm
from IA.models import SavedTrainingData, SavedModelsOLS

import statsmodels.api as sm
import matplotlib.pyplot as plt
from PIL import Image
import io
import requests


# Create your views here.
def Createmodel(request):
    return render(request, 'createModel.html')


class savedTraining(ListView):
    model = SavedTrainingData
    template_name = "trainingData.html"


def Index(request):
    return render(request, 'index.html')


def AllMLModels(request):
    return render(request, 'savedModels.html')


def create_ols_model(request):
    if request.method == 'POST':
        form = OLSForm(request.POST, request.FILES)
        if form.is_valid():
            model_ols = form.save(commit=False)
            endpoint = form.cleaned_data['endpoint']
            response = requests.get(endpoint)
            if response.status_code == 200:
                X_data = np.array([float(x) for x in response.json()])
                training_data = form.cleaned_data['training_data'].values
                Y_data = np.array([float(y) if y != '' else np.nan for y in eval(training_data)])
                print(len(X_data))
                print(len(Y_data))
                min_length = min(len(X_data), len(Y_data))
                X_data = X_data[:min_length]
                Y_data = Y_data[:min_length]
                if len(X_data) == len(Y_data):
                    X = sm.add_constant(X_data)
                    model = sm.OLS(Y_data, X).fit()
                    model_ols.values = model.summary().as_text()
                    if form.cleaned_data['generate_plot']:
                        fig, ax = plt.subplots()
                        ax.plot(X_data, Y_data, 'o', label="Data")
                        ax.plot(X_data, model.predict(X), 'r--.', label="OLS Prediction")
                        ax.legend()
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        image = Image.open(buf)
                        model_ols.plot.save("plot.png", ImageFile(buf), save=True)
                    model_ols.save()
                    return redirect('savedModels')
                else:
                    form.add_error(None, "El tamaño de los datos de X y Y no coincide.")
            else:
                form.add_error(None, "Error al obtener datos del endpoint.")
        return render(request, 'Methods/ols.html', {'form': form})
    else:
        form = OLSForm()
    return render(request, 'Methods/ols.html', {'form': form})


class OLSModels(ListView):
    model = SavedModelsOLS
    template_name = "models/linear/ols.html"

    def get_queryset(self):
        return SavedModelsOLS.objects.all()


class OLSModelDetail(DetailView):
    model = SavedModelsOLS
    template_name = 'models/linear/ols_detail.html'


def NewData(request):
    form = DataForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            name = form.cleaned_data['name']
            if 'csv_file' in request.FILES:
                csv_file = request.FILES['csv_file']
                decoded_file = csv_file.read().decode('utf-8').splitlines()
                reader = csv.reader(decoded_file)
                headers = next(reader)
                selected_column_index = int(request.POST.get('column_choice'))
                selected_column = [row[selected_column_index] for row in reader if
                                   row and row[selected_column_index].strip()]
                values = list(selected_column)
                saved_data = SavedTrainingData(name=name, values=values)
                saved_data.save()
            else:
                values = form.cleaned_data['values']
                saved_data = SavedTrainingData(name=name, values=values)
                saved_data.save()
            return redirect('trainingData')
    return render(request, 'trainingData/new.html', {'form': form})


def edit_data_view(request, pk):
    dato = get_object_or_404(SavedTrainingData, pk=pk)
    if request.method == 'POST':
        form = DataForm(request.POST, instance=dato)
        if form.is_valid():
            form.save()
            return redirect('trainingData')
    else:
        form = DataForm(instance=dato)
    return render(request, 'trainingData/editTrainingData.html', {'form': form})


@csrf_exempt
def delete_data_view(request):
    if request.method == 'POST':
        id = request.POST.get('id')
        item = SavedTrainingData.objects.get(pk=id)
        item.delete()
        return JsonResponse({'status': 'success'})
    return JsonResponse({'status': 'fail'})
