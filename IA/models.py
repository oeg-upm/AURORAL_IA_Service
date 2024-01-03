from django.db import models

# Create your models here.

class SavedTrainingData(models.Model):
    name = models.CharField(max_length=100)
    values = models.TextField()

    def __str__(self):
        return self.name

