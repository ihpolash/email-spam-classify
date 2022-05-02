from django.urls import path, include
from email_spam import views
from django.conf.urls.static import static

urlpatterns = [
    path('train_model/', views.TrainModelView.as_view(), name='trainmodel'),
    path('email_classify/', views.email_classify.as_view(), name='emailclassify'),
]