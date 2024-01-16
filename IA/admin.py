from django.contrib import admin
from django.contrib import admin
from .models import SavedTrainingData, SavedModelsOLS

# Register your models here.
admin.site.register(SavedTrainingData)
admin.site.register(SavedModelsOLS)