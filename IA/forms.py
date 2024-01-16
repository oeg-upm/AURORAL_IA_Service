from django.core.exceptions import ValidationError
from django.forms import ModelForm, ModelChoiceField, BooleanField

from IA.models import SavedTrainingData, SavedModelsOLS


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


class OLSForm(ModelForm):
    training_data = ModelChoiceField(queryset=SavedTrainingData.objects.all(), required=True, label="Training Data")
    generate_plot = BooleanField(required=False, label='Generate plot')

    class Meta:
        model = SavedModelsOLS
        fields = ['name', 'endpoint', 'plot']

    def __init__(self, *args, **kwargs):
        super(OLSForm, self).__init__(*args, **kwargs)
        self.fields['name'].widget.attrs.update({'class': 'form-control'})
        self.fields['endpoint'].widget.attrs.update({'class': 'form-control'})
        self.fields['training_data'].widget.attrs.update({'class': 'form-control'})
        self.fields['generate_plot'].widget.attrs.update({'class': 'form-check-input'})

    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data
