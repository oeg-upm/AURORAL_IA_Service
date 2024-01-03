from IA import views
from django.urls import path

from IA.views import edit_data_view, delete_data_view

urlpatterns = [
    path("", views.Index, name="index"),
    path("createModel", views.Createmodel, name="createModel"),
    path("savedModels", views.AllMLModels, name="savedModels"),
    path("trainingData", views.savedTraining.as_view(), name="trainingData"),
    path("trainingData/new", views.NewData, name="trainingData/new"),
    path('trainingData/edit/<int:pk>/', edit_data_view, name='trainingData/editTrainingData'),
    path('delete/', delete_data_view, name='delete_data'),
]