from django.urls import path
from nlp_model_analysis import views

urlpatterns = [
    path('', views.nlp_model_analysis, name='nlp_model_analysis'),
    #path("<str:mn>/", views.bias_metrics, name="bias_metrics"),
    path("gender_bias_metrics", views.gender_bias_metrics, name="gender_bias_metrics"),
    path("racial_bias_metrics", views.racial_bias_metrics, name="racial_bias_metrics"),
    path("query_sentence", views.query_sentence, name="query_sentence"),
]
