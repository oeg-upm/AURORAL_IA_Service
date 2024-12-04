from django.core.exceptions import ValidationError
from django.forms import ModelForm, ModelChoiceField, BooleanField, IntegerField, CheckboxSelectMultiple, \
    ModelMultipleChoiceField

from IA.models import SavedTrainingData, SavedModelsOLS, SavedModelsLasso, SavedModelsSVMClassification, \
    SavedModelsSVMRegression, SavedModelsKMeans


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


class LassoForm(ModelForm):
    training_data = ModelChoiceField(queryset=SavedTrainingData.objects.all(), required=True, label="Training Data")
    generate_plot = BooleanField(required=False, label='Generate plot')

    class Meta:
        model = SavedModelsLasso
        fields = ['name', 'endpoint', 'plot']

    def __init__(self, *args, **kwargs):
        super(LassoForm, self).__init__(*args, **kwargs)
        self.fields['name'].widget.attrs.update({'class': 'form-control'})
        self.fields['endpoint'].widget.attrs.update({'class': 'form-control'})
        self.fields['training_data'].widget.attrs.update({'class': 'form-control'})
        self.fields['generate_plot'].widget.attrs.update({'class': 'form-check-input'})

    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data


class SVMClassificationForm(ModelForm):
    training_data = ModelChoiceField(queryset=SavedTrainingData.objects.all(), required=True, label="Training Data")
    generate_plot = BooleanField(required=False, label='Generate plot')

    class Meta:
        model = SavedModelsSVMClassification
        fields = ['name', 'endpoint', 'plot']

    def __init__(self, *args, **kwargs):
        super(SVMClassificationForm, self).__init__(*args, **kwargs)
        self.fields['name'].widget.attrs.update({'class': 'form-control'})
        self.fields['endpoint'].widget.attrs.update({'class': 'form-control'})
        self.fields['training_data'].widget.attrs.update({'class': 'form-control'})
        self.fields['generate_plot'].widget.attrs.update({'class': 'form-check-input'})

    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data


class SVMRegressionForm(ModelForm):
    training_data = ModelChoiceField(queryset=SavedTrainingData.objects.all(), required=True, label="Training Data")
    generate_plot = BooleanField(required=False, label='Generate plot')

    class Meta:
        model = SavedModelsSVMRegression
        fields = ['name', 'endpoint', 'plot']

    def __init__(self, *args, **kwargs):
        super(SVMRegressionForm, self).__init__(*args, **kwargs)
        self.fields['name'].widget.attrs.update({'class': 'form-control'})
        self.fields['endpoint'].widget.attrs.update({'class': 'form-control'})
        self.fields['training_data'].widget.attrs.update({'class': 'form-control'})
        self.fields['generate_plot'].widget.attrs.update({'class': 'form-check-input'})

    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data


class KMeansForm(ModelForm):
    num_clusters = IntegerField(min_value=2, initial=3, label="Number of Clusters")
    generate_plot = BooleanField(required=False, label='Generate plot')
    datasets = ModelMultipleChoiceField(
        queryset=SavedTrainingData.objects.all(),
        widget=CheckboxSelectMultiple,
        required=True,
        label="Training Datasets"
    )
    class Meta:
        model = SavedModelsKMeans
        fields = ['name', 'endpoint', 'datasets', 'num_clusters', 'plot']

    def __init__(self, *args, **kwargs):
        super(KMeansForm, self).__init__(*args, **kwargs)
        self.fields['name'].widget.attrs.update({'class': 'form-control'})
        self.fields['endpoint'].widget.attrs.update({'class': 'form-control'})
        self.fields['num_clusters'].widget.attrs.update({'class': 'form-control'})
        self.fields['datasets'].widget.attrs.update({'class': 'form-control'})
        self.fields['generate_plot'].widget.attrs.update({'class': 'form-check-input'})

    def clean(self):
        cleaned_data = super().clean()
        return cleaned_data