import csv

from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import ListView

from IA.forms import DataForm
from IA.models import SavedTrainingData


# Create your views here.
def Createmodel(request):
    return render(request, 'createModel.html')

class savedTraining(ListView):
    model = SavedTrainingData
    template_name = "trainingData.html"

def Index(request):
    return render(request, 'index.html')

def AllMLModels(request):
    return render(request,'savedModels.html')


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
                selected_column = [row[selected_column_index] for row in reader if row and row[selected_column_index].strip()]
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