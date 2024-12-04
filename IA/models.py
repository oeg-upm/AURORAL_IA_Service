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


class SavedModelsLasso(models.Model):
    name = models.CharField(max_length=100)
    plot = models.ImageField(upload_to='static/lasso/', null=True, blank=True)
    endpoint = models.CharField(max_length=400)
    values = models.TextField()

    def __str__(self):
        return self.name


class SavedModelsSVMClassification(models.Model):
    name = models.CharField(max_length=100)
    plot = models.ImageField(upload_to='static/svm_classification/', null=True, blank=True)
    endpoint = models.CharField(max_length=400)
    values = models.TextField()

    def __str__(self):
        return self.name


class SavedModelsSVMRegression(models.Model):
    name = models.CharField(max_length=100)
    plot = models.ImageField(upload_to='static/svm_regression/', null=True, blank=True)
    endpoint = models.CharField(max_length=400)
    values = models.TextField()

    def __str__(self):
        return self.name


class SavedModelsKMeans(models.Model):
    name = models.CharField(max_length=100)
    plot = models.ImageField(upload_to='static/kmeans/', null=True, blank=True)
    datasets = models.ManyToManyField(SavedTrainingData)
    endpoint = models.CharField(max_length=400)
    values = models.TextField()

    def __str__(self):
        return self.name