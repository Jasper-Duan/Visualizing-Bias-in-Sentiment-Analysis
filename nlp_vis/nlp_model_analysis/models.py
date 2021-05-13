from django.db import models

# Create your models here.
class nlp_info(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    path = models.TextField()

#image = models.FilePathField(path="/img")
#accuracy = models.CharField(max_length=20)