from django.http import HttpResponse,HttpResponseNotFound,HttpResponseRedirect
from .models import Patient
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from os import path
from glob import glob
import pyrebase
from zipfile import ZipFile
from .forms import UploadForm
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import plot
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.graph_objs import Scatter

# from django.core.mail import EmailMessage, send_mail
# # from geeksforgeeks import settings
# from django.contrib.sites.shortcuts import get_current_site
# from django.template.loader import render_to_string
# from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
# # from django.utils.encoding import force_bytes, force_text
from django.contrib.auth import authenticate, login, logout
# from . tokens import generate_token
# Create your views here.
def index(request):
    dashboard= Patient.objects.all()
    return render(request,'dashboard/index.html',
    {
        'dashboard':dashboard,
    }
    )
def details(request,patient_slug):
    selected_patient = Patient.objects.get(slug=patient_slug)
    form = UploadForm(request.POST,request.FILES)
    if request.method == 'POST':
        print(form)
        if form.is_valid():
            form.save()
            return render(request,'dashboard/details.html',{
            'patient_name' : selected_patient.name,
            'patient_bed' : selected_patient.bedno,
            'patient_id' : selected_patient.patientid,
            'patient_doctor' : selected_patient.doctorname,
            'patient_aadhar' : selected_patient.aadharno,
            'patient_device' : selected_patient.deviceid,
            'patient_gender' : selected_patient.gender,
            'patient_age' : selected_patient.age,
            'patient_blood' : selected_patient.bloodtype,
            'patient_allergies' : selected_patient.allergies,
            'patient_diseases' : selected_patient.diseases,
            'patient_height' : selected_patient.height,
            'patient_weight' : selected_patient.weight,
            'patient_color' : selected_patient.color,
            'form':form
        })
    return render(request,'dashboard/details.html',{
        'patient_name' : selected_patient.name,
        'patient_bed' : selected_patient.bedno,
        'patient_id' : selected_patient.patientid,
        'patient_doctor' : selected_patient.doctorname,
        'patient_aadhar' : selected_patient.aadharno,
        'patient_device' : selected_patient.deviceid,
        'patient_gender' : selected_patient.gender,
        'patient_age' : selected_patient.age,
        'patient_blood' : selected_patient.bloodtype,
        'patient_allergies' : selected_patient.allergies,
        'patient_diseases' : selected_patient.diseases,
        'patient_height' : selected_patient.height,
        'patient_weight' : selected_patient.weight,
        'patient_color' : selected_patient.color,
        'form':form
    })


def home(request):
    return render(request,'dashboard/home.html')

def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']

        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('home')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('home')

        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('home')

        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('home')

        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric!!")
            return redirect('home')

        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        # myuser.is_active = False
        myuser.is_active = False
        myuser.save()
        messages.success(request, "Your Account has been created succesfully!! Please check your email to confirm your email address in order to activate your account.")

        # # Welcome Email
        # subject = "Welcome to GFG- Django Login!!"
        # message = "Hello " + myuser.first_name + "!! \n" + "Welcome to GFG!! \nThank you for visiting our website\n. We have also sent you a confirmation email, please confirm your email address. \n\nThanking You\nAnubhav Madhav"
        # from_email = settings.EMAIL_HOST_USER
        # to_list = [myuser.email]
        # send_mail(subject, message, from_email, to_list, fail_silently=True)

        # # Email Address Confirmation Email
        # current_site = get_current_site(request)
        # email_subject = "Confirm your Email @ GFG - Django Login!!"
        # message2 = render_to_string('email_confirmation.html',{

        #     'name': myuser.first_name,
        #     'domain': current_site.domain,
        #     'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
        #     'token': generate_token.make_token(myuser)
        # })
        # email = EmailMessage(
        # email_subject,
        # message2,
        # settings.EMAIL_HOST_USER,
        # [myuser.email],
        # )
        # email.fail_silently = True
        # email.send()

        return redirect('signin')


    return render(request, "dashboard/signup.html")


# def activate(request,uidb64,token):
#     try:
#         uid = force_text(urlsafe_base64_decode(uidb64))
#         myuser = User.objects.get(pk=uid)
#     except (TypeError,ValueError,OverflowError,User.DoesNotExist):
#         myuser = None

#     if myuser is not None and generate_token.check_token(myuser,token):
#         myuser.is_active = True
#         # user.profile.signup_confirmation = True
#         myuser.save()
#         login(request,myuser)
#         messages.success(request, "Your Account has been activated!!")
#         return redirect('signin')
#     else:
#         return render(request,'activation_failed.html')


def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']

        user = authenticate(username=username, password=pass1)

        if user is not None:
            login(request, user)
            fname = user.first_name
            # messages.success(request, "Logged In Sucessfully!!")
            return redirect('dashboard/')
        else:
            messages.error(request, "Bad Credentials!!")
            return redirect('home')

    return render(request, 'dashboard/signin.html')


def signout(request):
    logout(request)
    messages.success(request, "Logged Out Successfully!!")
    return redirect('home')

def pdf_view(request):
    fs = FileSystemStorage()
    filenames = glob(path.join('media',"*.pdf"))
    zipObj = ZipFile('media/Prescription.zip', 'w')
    for filename in filenames:
        zipObj.write(str(filename))
    filename = str(path.join('media',"Prescription.zip"))[6:]

    if fs.exists(filename):
        with fs.open(filename) as pdf:
            response = HttpResponse(open(path.join('media',"Prescription.zip"), 'rb').read())
            response['Content-Type'] = 'application/x-zip-compressed'
            response['Content-Disposition'] = 'attachment; filename=Prescription_Download.zip'
            return response
    else:
        return HttpResponseNotFound('The requested pdf was not found in our server.')


def line_chart(request):
    config = {
    'apiKey': "AIzaSyAMzyV5TrnaxZxhONkEzb1UArSlTpxR-II",
    'authDomain': "capstonetest-2a851.firebaseapp.com",
    'projectId': "capstonetest-2a851",
    'databaseURL':"https://capstonetest-2a851-default-rtdb.firebaseio.com/",
    'storageBucket': "http://capstonetest-2a851.appspot.com/",
    'messagingSenderId': "31150681068",
    'appId': "1:31150681068:web:6bc219e86d7e6bdfaf6ec9",
    'measurementId': "G-71ETB9N43N"
    };
    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    a = database.get()
    b = list(a.val().items())[0][1]
    data = list(b.values())

    value = []
    value2 = []
    value3 = []
    value4 = []

    for i in data:
      value.append(int(i.split(',')[0]))
      value2.append(float(i.split(',')[3].split('\\')[0]))
      value3.append(int(i.split(',')[1]))
      value4.append(float(i.split(',')[2]))

    df = pd.DataFrame(columns=['Time','Blood Pressure','Temperature','Oxygen Saturation','Blood Glucose'])
    df['Time'] = b.keys()
    df2 = pd.DataFrame(a.val().values())
    df['Blood Pressure'] = value
    df['Temperature'] = value2
    df['Oxygen Saturation'] = value3
    df['Blood Glucose'] = value4
    df['Time'] = pd.to_datetime(df['Time'])
    d = datetime.datetime.now() - datetime.timedelta(hours=24)
    df = df[~(df['Time'] < str(d))]
    plot_div = plot([Scatter(x=df['Time'], y=df['Blood Pressure'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div1 = plot([Scatter(x=df['Time'], y=df['Temperature'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div2 = plot([Scatter(x=df['Time'], y=df['Oxygen Saturation'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div3 = plot([Scatter(x=df['Time'], y=df['Blood Glucose'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    return render(request, "dashboard/pastexp.html", context={'plot_div': plot_div,'plot_div1':plot_div1,'plot_div2':plot_div2,'plot_div3':plot_div3})

def line_chart1(request):
    config = {
    'apiKey': "AIzaSyAMzyV5TrnaxZxhONkEzb1UArSlTpxR-II",
    'authDomain': "capstonetest-2a851.firebaseapp.com",
    'projectId': "capstonetest-2a851",
    'databaseURL':"https://capstonetest-2a851-default-rtdb.firebaseio.com/",
    'storageBucket': "http://capstonetest-2a851.appspot.com/",
    'messagingSenderId': "31150681068",
    'appId': "1:31150681068:web:6bc219e86d7e6bdfaf6ec9",
    'measurementId': "G-71ETB9N43N"
    };
    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    a = database.get()
    b = list(a.val().items())[0][1]
    data = list(b.values())

    value = []
    value2 = []
    value3 = []
    value4 = []

    for i in data:
      value.append(int(i.split(',')[0]))
      value2.append(float(i.split(',')[3].split('\\')[0]))
      value3.append(int(i.split(',')[1]))
      value4.append(float(i.split(',')[2]))

    df = pd.DataFrame(columns=['Time','Blood Pressure','Temperature','Oxygen Saturation','Blood Glucose'])
    df['Time'] = b.keys()
    df2 = pd.DataFrame(a.val().values())
    df['Blood Pressure'] = value
    df['Temperature'] = value2
    df['Oxygen Saturation'] = value3
    df['Blood Glucose'] = value4
    df['Time'] = pd.to_datetime(df['Time'])
    d = datetime.datetime.now() - datetime.timedelta(hours=48)
    df = df[~(df['Time'] < str(d))]
    plot_div = plot([Scatter(x=df['Time'], y=df['Blood Pressure'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div1 = plot([Scatter(x=df['Time'], y=df['Temperature'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div2 = plot([Scatter(x=df['Time'], y=df['Oxygen Saturation'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div3 = plot([Scatter(x=df['Time'], y=df['Blood Glucose'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    return render(request, "dashboard/pastexp1.html", context={'plot_div': plot_div,'plot_div1':plot_div1,'plot_div2':plot_div2,'plot_div3':plot_div3})

def line_chart2(request):
    config = {
    'apiKey': "AIzaSyAMzyV5TrnaxZxhONkEzb1UArSlTpxR-II",
    'authDomain': "capstonetest-2a851.firebaseapp.com",
    'projectId': "capstonetest-2a851",
    'databaseURL':"https://capstonetest-2a851-default-rtdb.firebaseio.com/",
    'storageBucket': "http://capstonetest-2a851.appspot.com/",
    'messagingSenderId': "31150681068",
    'appId': "1:31150681068:web:6bc219e86d7e6bdfaf6ec9",
    'measurementId': "G-71ETB9N43N"
    };
    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    a = database.get()
    b = list(a.val().items())[0][1]
    data = list(b.values())

    value = []
    value2 = []
    value3 = []
    value4 = []

    for i in data:
      value.append(int(i.split(',')[0]))
      value2.append(float(i.split(',')[3].split('\\')[0]))
      value3.append(int(i.split(',')[1]))
      value4.append(float(i.split(',')[2]))

    df = pd.DataFrame(columns=['Time','Blood Pressure','Temperature','Oxygen Saturation','Blood Glucose'])
    df['Time'] = b.keys()
    df2 = pd.DataFrame(a.val().values())
    df['Blood Pressure'] = value
    df['Temperature'] = value2
    df['Oxygen Saturation'] = value3
    df['Blood Glucose'] = value4
    df['Time'] = pd.to_datetime(df['Time'])
    d = datetime.datetime.now() - datetime.timedelta(hours=72)
    df = df[~(df['Time'] < str(d))]
    plot_div = plot([Scatter(x=df['Time'], y=df['Blood Pressure'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div1 = plot([Scatter(x=df['Time'], y=df['Temperature'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div2 = plot([Scatter(x=df['Time'], y=df['Oxygen Saturation'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div3 = plot([Scatter(x=df['Time'], y=df['Blood Glucose'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    return render(request, "dashboard/pastexp2.html", context={'plot_div': plot_div,'plot_div1':plot_div1,'plot_div2':plot_div2,'plot_div3':plot_div3})

# def read_file(request):
#     f = open(r'C:/Users/SAHAJ/Desktop/Project/Environmentse/env/src/healthdash/alert.txt', 'r')
#     file_contents = f.read()
#     f.close()
#     args = {'result' : file_contents }
#     return render(request, "alert.html", context, args)

def read_file(request):
    file1 = open(r'alert.txt',"r",encoding='UTF-8')
    d = file1.read()
    print(d) #prints on cmd in the same formatting as in text file
    return render(request,'dashboard/alert.html',{'dat':d})

def admin(request):
    return render(request,'admin')


def future_graph(request):
    config = {
    'apiKey': "AIzaSyAGLHV0lJ0qhPmcoVO3HulVPwfRf0OpJxo",
    'authDomain': "capstonefuture.firebaseapp.com",
    'databaseURL': "https://capstonefuture-default-rtdb.firebaseio.com",
    'projectId': "capstonefuture",
    'storageBucket': "capstonefuture.appspot.com",
    'messagingSenderId': "609555345653",
    'appId': "1:609555345653:web:9049fb485f8846474715bc",
    'measurementId': "G-0VBHKNTPCT"
    };
    firebase = pyrebase.initialize_app(config)
    database = firebase.database()
    a = database.get()
    print()
    b = list(a.val().items())

    value = []
    value2 = []
    value3 = []
    value4 = []
    time = []

    for i in b:
        key = i[1].keys()
        val = i[1].values()
        time.append(str(list(key)[0]))
        val = str(list(val)[0])
        value.append(int(val.split(',')[0]))
        value2.append(float(val.split(',')[3].split('\\')[0]))
        value3.append(int(val.split(',')[1]))
        value4.append(float(val.split(',')[2]))
        print(value,value2,value3,value4)

    df = pd.DataFrame(columns=['Time','Blood Pressure','Temperature','Oxygen Saturation','Blood Glucose'])
    df['Time'] = time
    df2 = pd.DataFrame(a.val().values())
    df['Blood Pressure'] = value
    df['Temperature'] = value2
    df['Oxygen Saturation'] = value3
    df['Blood Glucose'] = value4
    df['Time'] = pd.to_datetime(df['Time'])
    # d = datetime.datetime.now() - datetime.timedelta(hours=24)
    # df = df[~(df['Time'] < str(d))]
    plot_div = plot([Scatter(x=df['Time'], y=df['Blood Pressure'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div1 = plot([Scatter(x=df['Time'], y=df['Temperature'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div2 = plot([Scatter(x=df['Time'], y=df['Oxygen Saturation'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    plot_div3 = plot([Scatter(x=df['Time'], y=df['Blood Glucose'],mode='lines', name='test',opacity=0.8, marker_color='green')],output_type='div')
    return render(request, "dashboard/future.html", context={'plot_div': plot_div,'plot_div1':plot_div1,'plot_div2':plot_div2,'plot_div3':plot_div3})
