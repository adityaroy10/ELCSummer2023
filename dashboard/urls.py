from unicodedata import name
from django.urls import path,re_path
from . import views
from django.contrib import admin
from django.conf.urls import url

urlpatterns = [
    path('',views.home,name='home'),
    path('admin/',views.admin,name='admin'),
    path('signup',views.signup,name='signup'),
    path('signin',views.signin,name='signin'),
    path('signout',views.signout,name='signout'),
    path('dashboard/',views.index,name='dashboard'),
    path('dashboard/<slug:patient_slug>',views.details,name='patient'),
    re_path(r'pdf/$', views.pdf_view, name='pdf_view'),
    path('alertlog/',views.read_file,name='alertlog'),
    path('dashboard/past/', views.line_chart, name='line_chart'),
    path('dashboard/past1/', views.line_chart1, name='line_chart2'),
    path('dashboard/past2/', views.line_chart2, name='line_chart3'),
    path('dashboard/future/', views.future_graph, name='future_graph'),
]
