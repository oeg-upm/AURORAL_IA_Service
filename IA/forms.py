from django.core.exceptions import ValidationError
from django.forms import ModelForm, forms

from IA.models import SavedTrainingData


class DataForm(ModelForm):


    class Meta:
        model = SavedTrainingData
        fields = ['name', 'values']

    def __init__(self, *args, **kwargs):
        super(DataForm, self).__init__(*args, **kwargs)
        for field in self.fields:
            self.fields[field].widget.attrs['class'] = 'form-control'
        self.fields['values'].required = False

    def clean_name(self):
        name = self.cleaned_data['name']
        if SavedTrainingData.objects.filter(name=name).exists():
            raise ValidationError("Ya existe un elemento con este nombre.")
        return name