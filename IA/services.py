from django.http import JsonResponse
from sklearn import linear_model, svm, cluster
import numpy as np

def linear_ols(data):
    X = np.array(data['features'])
    y = np.array(data['target'])
    model = linear_model.LinearRegression()
    model.fit(X, y)
    return JsonResponse({'coef': model.coef_.tolist(), 'intercept': model.intercept_})

def linear_lasso(data):
    X = np.array(data['features'])
    y = np.array(data['target'])
    model = linear_model.Lasso()
    model.fit(X, y)
    return JsonResponse({'coef': model.coef_.tolist(), 'intercept': model.intercept_})

def svm_classification(data):
    X = np.array(data['features'])
    y = np.array(data['target'])
    model = svm.SVC()
    model.fit(X, y)
    return JsonResponse({'support_vectors': model.support_vectors_.tolist()})

def svm_regression(data):
    X = np.array(data['features'])
    y = np.array(data['target'])
    model = svm.SVR()
    model.fit(X, y)
    return JsonResponse({'support_vectors': model.support_vectors_.tolist()})

def kmeans(data, n_clusters, n_init):
    #X = np.array(data['features'])
    #n_clusters = request.args.get('n_clusters', 3, type=int)
    #n_init = request.args.get('n_init', 10, type=int)
    try:
        n_clusters = int(n_clusters)
    except ValueError:
        return JsonResponse({'error': 'Invalid n_clusters value'}), 400
    model = cluster.KMeans(n_clusters=n_clusters, n_init=n_init)
    #model.fit(X)
    return JsonResponse({'labels': model.labels_.tolist(), 'cluster_centers': model.cluster_centers_.tolist()})
