from django.shortcuts import render,redirect
from django.http import HttpResponse,HttpResponseRedirect

from analyzer.Pixel_robustness import Pixel_main
from .FGSM_robustness import FGSM_main
from analyzer.DeepFool_robustness import DeepFool_main
from analyzer.PDG_robustness import PGD_main
import os
import webbrowser
  
dataset = None
model = None
attack1 = False
attack2 = False
attack3 = False
attack4 = False
defence1 = False


def index(request):
    print(os.getcwd())
    global attack1,attack2,attack3,attack4,defence1,dataset
    if request.method == "POST":
        data = dict(request.POST.lists())
        dataset = request.POST['model']
        attack = data.get('attack')
        print(attack[0])
        if attack[0] == 'attack1':
            attack1=True
        if attack[0] == 'attack2':
            attack2=True
        if attack[0] == 'attack3':
            attack3=True
        if attack[0] == 'attack4':
            attack4=True
        if(data.get("defence1")!=None):
            defence1 = True
        listofops = loading()
        print(listofops)
        return render(request,"loading.html",{"operations":listofops})
    else:
        attack1 = False
        attack2 = False
        attack3 = False
        attack4 = False
        defence1 = False
    
        return render(request,"index.html")

def loading():
    listofops = []
    global attack1,attack2,attack3,attack4,defence1,dataset
    if attack1:
        listofops.append("Fast Gradient Sign Method")
    else:
        listofops.append("")
    if attack2:
        listofops.append("Pixel Attack")
    else:
        listofops.append("")
    if attack3:
        listofops.append("Project Gradient Descent")
    else:
        listofops.append("")
    if attack4:
        listofops.append("DeepFool")
    else:
        listofops.append("")
    if defence1:
        listofops.append("Adversarial Training")
    else:
        listofops.append("")
    listofops.append(dataset)
    
    return listofops

def processing(request):
    global attack1,attack2,attack3,attack4,defence1,dataset
    if(attack1):
            #call their functions
            print('FGSM has started')
            FGSM_main(dataset)
            
    if(attack2):
            #call their functions
            print('Pixel has started')
            Pixel_main(dataset)
            #save the returned efficiency in the database
            
    if(attack3):
            #call their functions
            print('PGD has started')
            PGD_main(dataset)
    
    if(attack4):
            #call their functions
            print('DeepFool has started')
            DeepFool_main(dataset)
            
   
   
    print("operation success")
    
    currentpath = os.getcwd()
    path = currentpath+'\\report.html'
    webbrowser.open(path,new=2)
    return HttpResponseRedirect("/")


