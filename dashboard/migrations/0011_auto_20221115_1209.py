# Generated by Django 3.2 on 2022-11-15 06:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0010_patient_diseases'),
    ]

    operations = [
        migrations.AddField(
            model_name='patient',
            name='height',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='patient',
            name='weight',
            field=models.IntegerField(default=0),
        ),
    ]