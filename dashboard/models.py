from django.db import models
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
# Create your models here.

Gender_choice=(
    ('Male','MALE'),
    ('Female','FEMALE'),
    ('Others','OTHERS')
    )

BloodType_choice=(
    ('A+','A+'),
    ('A-','A-'),
    ('B+','B+'),
    ('B-','B-'),
    ('O+','O+'),
    ('O-','O-'),
    ('AB+','AB+'),
    ('AB-','AB-'),
    )

class Patient(models.Model):
    bedno = models.IntegerField()
    patientid = models.BigIntegerField(unique=True)
    name = models.CharField(max_length=200)
    gender = models.CharField(max_length=6,choices=Gender_choice,default='Male')
    color = models.IntegerField(default=1)
    age = models.IntegerField(default=0)
    bloodtype=models.CharField(max_length=3,choices=BloodType_choice,default='A+')
    allergies=models.CharField(max_length=100,default='No Allergies')
    diseases=models.CharField(max_length=100,default='No Diseases')
    height = models.IntegerField(default=0)
    weight = models.IntegerField(default=0)
    doctorname = models.CharField(max_length=200)
    aadharno = models.PositiveBigIntegerField()
    deviceid = models.IntegerField(unique=True)
    slug = models.SlugField(unique=True)

class pdf_upload(models.Model):
    title = models.CharField(max_length = 80,null=True)
    pdf = models.FileField(storage=FileSystemStorage(location=settings.MEDIA_ROOT))

    class Meta:
        ordering = ['title']

    def __str__(self):
        return f"{self.title}"
