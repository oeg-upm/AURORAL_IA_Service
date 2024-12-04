from IA import views
from django.urls import path

from IA.views import edit_data_view, delete_data_view, OLSModelDetail, LassoModelDetail, SVMClassificationModelDetail, \
    SVMRegressionModelDetail, KMeansModelDetail

urlpatterns = [
    path("", views.AllMLModels, name="index"),
    path("createModel", views.Createmodel, name="createModel"),
    path("savedModels", views.AllMLModels, name="savedModels"),
    path("trainingData", views.savedTraining.as_view(), name="trainingData"),
    path("trainingData/new", views.NewData, name="trainingData/new"),
    path('trainingData/edit/<int:pk>/', edit_data_view, name='trainingData/editTrainingData'),
    path('delete/', delete_data_view, name='delete_data'),
    path('methods/linear/ols', views.create_ols_model, name='Methods/ols'),
    path('methods/linear/lasso', views.create_lasso_model, name='Methods/lasso'),
    path('methods/svm/classification', views.create_svm_classification_model, name='Methods/classification'),
    path('methods/svm/regression', views.create_svm_regression_model, name='Methods/regression'),
    path('methods/cluster/kmeans', views.create_kmeans_model, name='Methods/kmeans'),
    path('models/linear/ols', views.OLSModels.as_view(), name='models/ols'),
    path('models/ols/<int:pk>/', OLSModelDetail.as_view(), name='ols_model_detail'),
    path('models/linear/lasso', views.LassoModels.as_view(), name='models/lasso'),
    path('models/lasso/<int:pk>/', LassoModelDetail.as_view(), name='lasso_model_detail'),
    path('models/svm/classification', views.SVMClassificationModels.as_view(), name='models/classification'),
    path('models/classification/<int:pk>/', SVMClassificationModelDetail.as_view(), name='classification_model_detail'),
    path('models/svm/regression', views.SVMRegressionModels.as_view(), name='models/regression'),
    path('models/regression/<int:pk>/', SVMRegressionModelDetail.as_view(), name='regression_model_detail'),
    path('models/kmeans', views.KMeansModels.as_view(), name='models/kmeans'),
    path('models/kmeans/<int:pk>/', KMeansModelDetail.as_view(), name='kmeans_model_detail'),
]