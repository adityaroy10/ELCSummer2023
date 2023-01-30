from .models import pdf_upload
from django import forms

class UploadForm(forms.ModelForm):
    class Meta:
        model = pdf_upload
        fields = ('title', 'pdf',)
