import csv

import numpy as np
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView, DetailView
from django.core.files.images import ImageFile
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, SVR

from IA.forms import DataForm, OLSForm, LassoForm, SVMClassificationForm, SVMRegressionForm, KMeansForm
from IA.models import SavedTrainingData, SavedModelsOLS, SavedModelsLasso, SavedModelsSVMClassification, \
    SavedModelsSVMRegression, SavedModelsKMeans

import statsmodels.api as sm
import matplotlib.pyplot as plt
from PIL import Image
import io
import requests
import json

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


def create_lasso_model(request):
    if request.method == 'POST':
        form = LassoForm(request.POST, request.FILES)
        if form.is_valid():
            model_lasso = form.save(commit=False)
            endpoint = form.cleaned_data['endpoint']
            response = requests.get(endpoint)
            if response.status_code == 200:
                X_data = np.array([float(x) for x in response.json()])
                training_data = form.cleaned_data['training_data'].values
                Y_data = np.array([float(y) if y != '' else np.nan for y in eval(training_data)])
                min_length = min(len(X_data), len(Y_data))
                X_data = X_data[:min_length]
                Y_data = Y_data[:min_length]
                if len(X_data) == len(Y_data):
                    lasso = Lasso(alpha=1.0)
                    lasso.fit(X_data.reshape(-1, 1), Y_data)
                    coefficients = lasso.coef_
                    intercept = lasso.intercept_
                    model_lasso.values = f"Coefficients: {coefficients}, Intercept: {intercept}"
                    if form.cleaned_data['generate_plot']:
                        fig, ax = plt.subplots()
                        ax.plot(X_data, Y_data, 'o', label="Data")
                        X_fit = np.linspace(min(X_data), max(X_data), 100)
                        Y_pred = lasso.predict(X_fit.reshape(-1, 1))
                        ax.plot(X_fit, Y_pred, 'r--', label="Lasso Prediction")
                        ax.legend()
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        image = Image.open(buf)
                        model_lasso.plot.save("plot.png", ImageFile(buf), save=True)
                    model_lasso.save()
                    return redirect('savedModels')
                else:
                    form.add_error(None, "El tamaño de los datos de X y Y no coincide.")
            else:
                form.add_error(None, "Error al obtener datos del endpoint.")
        return render(request, 'Methods/lasso.html', {'form': form})
    else:
        form = LassoForm()
    return render(request, 'Methods/lasso.html', {'form': form})


class LassoModels(ListView):
    model = SavedModelsLasso
    template_name = "models/lasso/lasso.html"

    def get_queryset(self):
        return SavedModelsLasso.objects.all()


class LassoModelDetail(DetailView):
    model = SavedModelsLasso
    template_name = 'models/lasso/lasso_detail.html'


def create_svm_classification_model(request):
    if request.method == 'POST':
        form = SVMClassificationForm(request.POST, request.FILES)
        if form.is_valid():
            model_svm = form.save(commit=False)
            endpoint = form.cleaned_data['endpoint']
            response = requests.get(endpoint)
            if response.status_code == 200:
                X_data = np.array([float(x) for x in response.json()])
                training_data = form.cleaned_data['training_data'].values
                Y_data = np.array([float(y) if y != '' else np.nan for y in eval(training_data)])
                min_length = min(len(X_data), len(Y_data))
                X_data = X_data[:min_length]
                Y_data = Y_data[:min_length]
                if len(X_data) == len(Y_data):
                    svm = SVC()
                    svm.fit(X_data.reshape(-1, 1), Y_data)
                    model_svm.values = str(svm.support_vectors_)
                    if form.cleaned_data['generate_plot']:
                        fig, ax = plt.subplots()
                        classes = np.unique(Y_data)
                        for cls in classes:
                            idx = np.where(Y_data == cls)
                            ax.scatter(X_data[idx], Y_data[idx], label=f'Class {cls}')
                        ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100, facecolors='none',
                                   edgecolors='k', label='Support Vectors')
                        ax.legend()
                        ax.set_title('SVM Classification')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        image = Image.open(buf)
                        model_svm.plot.save(f"{model_svm.name}_plot.png", ImageFile(buf), save=True)
                        buf.close()
                    model_svm.save()
                    return redirect('savedModels')
                else:
                    form.add_error(None, "El tamaño de los datos de X y Y no coincide.")
            else:
                form.add_error(None, "Error al obtener datos del endpoint.")
        return render(request, 'Methods/classification.html', {'form': form})
    else:
        form = SVMClassificationForm()
    return render(request, 'Methods/classification.html', {'form': form})


class SVMClassificationModels(ListView):
    model = SavedModelsSVMClassification
    template_name = "models/svm_classification/svm_classification.html"

    def get_queryset(self):
        return SavedModelsSVMClassification.objects.all()


class SVMClassificationModelDetail(DetailView):
    model = SavedModelsSVMClassification
    template_name = 'models/svm_classification/svm_classification_detail.html'


def create_svm_regression_model(request):
    if request.method == 'POST':
        form = SVMRegressionForm(request.POST, request.FILES)
        if form.is_valid():
            model_svm = form.save(commit=False)
            endpoint = form.cleaned_data['endpoint']
            response = requests.get(endpoint)
            if response.status_code == 200:
                X_data = np.array([float(x) for x in response.json()])
                training_data = form.cleaned_data['training_data'].values
                Y_data = np.array([float(y) if y != '' else np.nan for y in eval(training_data)])
                min_length = min(len(X_data), len(Y_data))
                X_data = X_data[:min_length]
                Y_data = Y_data[:min_length]
                if len(X_data) == len(Y_data):
                    svm = SVR()
                    svm.fit(X_data.reshape(-1, 1), Y_data)
                    model_svm.values = str(svm.support_)
                    if form.cleaned_data['generate_plot']:
                        fig, ax = plt.subplots()
                        X_fit = np.linspace(min(X_data), max(X_data), 100).reshape(-1, 1)
                        Y_pred = svm.predict(X_fit)
                        ax.scatter(X_data, Y_data, color='blue', label='Datos')
                        ax.plot(X_fit, Y_pred, color='red', label='Predicción SVM')
                        ax.set_title('Regresión SVM')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.legend()
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        image = Image.open(buf)
                        model_svm.plot.save(f"{model_svm.name}_plot.png", ImageFile(buf), save=True)
                        buf.close()
                    model_svm.save()
                    return redirect('savedModels')
                else:
                    form.add_error(None, "El tamaño de los datos de X y Y no coincide.")
            else:
                form.add_error(None, "Error al obtener datos del endpoint.")
        else:
            print(form.errors)
        return render(request, 'Methods/regression.html', {'form': form})
    else:
        form = SVMRegressionForm()
    return render(request, 'Methods/regression.html', {'form': form})


class SVMRegressionModels(ListView):
    model = SavedModelsSVMRegression
    template_name = "models/svm_regression/svm_regression.html"

    def get_queryset(self):
        return SavedModelsSVMRegression.objects.all()


class SVMRegressionModelDetail(DetailView):
    model = SavedModelsSVMRegression
    template_name = 'models/svm_regression/svm_regression_detail.html'


def create_kmeans_model(request):
    if request.method == 'POST':
        form = KMeansForm(request.POST, request.FILES)
        if form.is_valid():
            model_kmeans = form.save(commit=False)
            datasets = form.cleaned_data['datasets']
            training_data = []
            for dataset in datasets:
                training_data.extend([float(value) for value in eval(dataset.values)])
            endpoint = form.cleaned_data['endpoint']
            response = requests.get(endpoint)
            if response.status_code == 200:
                external_data = np.array([float(x) for x in response.json()])
                data = np.concatenate((training_data, external_data)).reshape(-1, 1)
                num_clusters = form.cleaned_data['num_clusters']
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(data)
                model_kmeans.values = str(kmeans.cluster_centers_)
                if form.cleaned_data['generate_plot']:
                    fig, ax = plt.subplots()
                    ax.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
                    centers = kmeans.cluster_centers_
                    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5, marker='x')
                    ax.set_title('K-Means Clustering')
                    ax.set_xlabel('X Axis')
                    ax.set_ylabel('Y Axis')
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    image = Image.open(buf)
                    model_kmeans.plot.save(f"{model_kmeans.name}_plot.png", ImageFile(buf), save=True)
                    buf.close()
                model_kmeans.save()
                return redirect('savedModels')
            else:
                form.add_error(None, "Error al obtener datos del endpoint.")
        else:
            print(form.errors)
        return render(request, 'Methods/kmeans.html', {'form': form})
    else:
        form = KMeansForm()
    return render(request, 'Methods/kmeans.html', {'form': form})


class KMeansModels(ListView):
    model = SavedModelsKMeans
    template_name = "models/kmeans/kmeans.html"

    def get_queryset(self):
        return SavedModelsKMeans.objects.all()


class KMeansModelDetail(DetailView):
    model = SavedModelsKMeans
    template_name = 'models/kmeans/kmeans_detail.html'


def NewData(request):
    form = DataForm(request.POST or None, request.FILES or None)
    if request.method == 'POST':
        if form.is_valid():
            name = form.cleaned_data['name']
            values = []

            # Código para manejar la carga de archivos CSV, si es necesario
            if 'csv_file' in request.FILES:
                csv_file = request.FILES['csv_file']
                decoded_file = csv_file.read().decode('utf-8').splitlines()
                reader = csv.reader(decoded_file)
                headers = next(reader)
                selected_column_index = int(request.POST.get('column_choice'))
                selected_column = [row[selected_column_index] for row in reader if
                                   row and row[selected_column_index].strip()]
                values = list(selected_column)

            # Código para manejar la carga de datos desde una URL que devuelve JSON
            elif 'csv_url' in request.POST and request.POST['csv_url'].strip():
                try:
                    csv_url = request.POST['csv_url']
                    response = requests.get(csv_url)
                    values = json.loads(response.content.decode('utf-8'))

                    # Verificar que la respuesta es una lista de strings
                    if not all(isinstance(item, str) for item in values):
                        raise ValueError("Invalid format: All items must be strings.")

                except Exception as e:
                    # Maneja errores al cargar o analizar JSON de la URL
                    # Este print es solo para desarrollo, considera enviar un mensaje de error al usuario
                    print(f"Error loading or parsing JSON from URL: {e}")
                    # Considera añadir un mensaje de error al contexto si quieres mostrarlo en el template
                    return render(request, 'trainingData/new.html',
                                  {'form': form, 'error': 'Invalid URL or data format.'})

            # Código para manejar los valores ingresados directamente en el formulario
            else:
                values = form.cleaned_data['values']

            if values:
                # Guarda los datos usando tu modelo SavedTrainingData
                saved_data = SavedTrainingData(name=name, values=values)
                saved_data.save()
                return redirect('trainingData')

    # Renderiza el formulario si no es POST o si hay errores
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
