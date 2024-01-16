from IA import views
from django.urls import path

from IA.views import edit_data_view, delete_data_view, OLSModelDetail

urlpatterns = [
    path("", views.Index, name="index"),
    path("createModel", views.Createmodel, name="createModel"),
    path("savedModels", views.AllMLModels, name="savedModels"),
    path("trainingData", views.savedTraining.as_view(), name="trainingData"),
    path("trainingData/new", views.NewData, name="trainingData/new"),
    path('trainingData/edit/<int:pk>/', edit_data_view, name='trainingData/editTrainingData'),
    path('delete/', delete_data_view, name='delete_data'),
    path('methods/linear/ols', views.create_ols_model, name='Methods/ols'),
    path('models/linear/lasso', delete_data_view, name='lasso'),
    path('models/svm/classification', delete_data_view, name='classification'),
    path('models/svm/regression', delete_data_view, name='delete_data'),
    path('models/cluster/kmeans', delete_data_view, name='kmeans'),
    path('models/linear/ols', views.OLSModels.as_view(), name='models/ols'),
    path('models/ols/<int:pk>/', OLSModelDetail.as_view(), name='ols_model_detail'),

]