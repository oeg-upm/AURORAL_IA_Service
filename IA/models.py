from django.db import models


# Create your models here.

class SavedTrainingData(models.Model):
    name = models.CharField(max_length=100)
    values = models.TextField()

    def __str__(self):
        return self.name


class SavedModelsOLS(models.Model):
    name = models.CharField(max_length=100)
    plot = models.ImageField(upload_to='static/ols/', null=True, blank=True)
    endpoint = models.CharField(max_length=400)
    values = models.TextField()

    def __str__(self):
        return self.name
