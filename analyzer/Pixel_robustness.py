import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from art.utils import load_dataset
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import PixelAttack
from art.defences.trainer import AdversarialTrainer
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf


dataset_type = {'mnist':'mnist_cnn_original.h5','cifar10':'cifar_resnet.h5'}

def load_data(dataset):
    (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(dataset)
    return (x_train, y_train, x_test, y_test, min_, max_)


def create_model(min_, max_,dataset):
    global dataset_type
    tf.compat.v1.disable_eager_execution()
    # path = get_file('mnist_cnn_original(1).h5', extract=False, path=config.ART_DATA_PATH,
    #             url='https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1')
    classifier_model = load_model(dataset_type[dataset])
    classifier = KerasClassifier(clip_values=(min_, max_), model=classifier_model, use_logits=False)
    return classifier

def create_robust_model(min_,max_,dataset):
    global dataset_type
    tf.compat.v1.disable_eager_execution()
    # path = get_file('mnist_cnn_original(1).h5', extract=False, path=config.ART_DATA_PATH,
    #             url='https://www.dropbox.com/s/p2nyzne9chcerid/mnist_cnn_original.h5?dl=1')
    robust_model = load_model(dataset_type[dataset])
    robust_classifier = KerasClassifier(clip_values=(min_, max_), model=robust_model, use_logits=False)
    return robust_classifier

def original_accuracy(classifier,x_test,y_test):
    x_test_pred = np.argmax(classifier.predict(x_test[:100]), axis=1)
    nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:100], axis=1))
    print("Original test data (first 100 images):")
    print("Correctly classified: {}".format(nb_correct_pred))
    print("Incorrectly classified: {}".format(100-nb_correct_pred))
    return (x_test_pred,nb_correct_pred)

def attack(classifier,x_test):
    attacker = PixelAttack(classifier, th = 4, es = 1, max_iter= 5, targeted = False, verbose = True)
    x_test_adv = attacker.generate(x_test[:100])
    return (x_test_adv, attacker)

def attacked_accuracy(classifier, x_test_adv,y_test):
    x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
    nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(y_test[:100], axis=1))
    print("Adversarial test data (first 100 images):")
    print("Correctly classified: {}".format(nb_correct_adv_pred))
    print("Incorrectly classified: {}".format(100-nb_correct_adv_pred))
    print(x_test_adv_pred)
    return (x_test_adv_pred, nb_correct_adv_pred)

def attack_robust(robust_classifier,x_train,y_train):
    attacker_robust= PixelAttack(robust_classifier, th = None, es = 1, max_iter= 5, targeted = False, verbose = False)
    trainer = AdversarialTrainer(robust_classifier, attacker_robust, ratio=1.0)
    trainer.fit(x_train[:100], y_train, nb_epochs=8, batch_size=50)
    return attacker_robust

def robust_with_attack(robust_classifier,x_test,y_test):
    x_test_robust_pred = np.argmax(robust_classifier.predict(x_test[:100]), axis=1)
    nb_correct_robust_pred = np.sum(x_test_robust_pred == np.argmax(y_test[:100], axis=1))
    print("Original test data (first 100 images):")
    print("Correctly classified: {}".format(nb_correct_robust_pred))
    print("Incorrectly classified: {}".format(100-nb_correct_robust_pred))
    return nb_correct_robust_pred
    #call attack() after this

def robust_accuracy(robust_classifier,x_test,y_test):
    x_test_adv_robust, cls = attack(robust_classifier,x_test)
    x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
    nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred == np.argmax(y_test[:100], axis=1))
    print("Adversarial test data (first 100 images):")
    print("Correctly classified: {}".format(nb_correct_adv_robust_pred))
    print("Incorrectly classified: {}".format(100-nb_correct_adv_robust_pred))
    return (nb_correct_adv_robust_pred,x_test_adv_robust)


def evaluate(attacker,attacker_robust,classifier,robust_classifier,nb_correct_pred,nb_correct_robust_pred,x_test,y_test):
    eps_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    nb_correct_original = []
    nb_correct_robust = []

    for eps in eps_range:
        attacker.set_params(**{'eps': eps})
        attacker_robust.set_params(**{'eps': eps})
        x_test_adv = attacker.generate(x_test[:100])
        x_test_adv_robust = attacker_robust.generate(x_test[:100])
    
        x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
        nb_correct_original += [np.sum(x_test_adv_pred == np.argmax(y_test[:100], axis=1))]
    
        x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
        nb_correct_robust += [np.sum(x_test_adv_robust_pred == np.argmax(y_test[:100], axis=1))]

    eps_range = [0] + eps_range
    nb_correct_original = [nb_correct_pred] + nb_correct_original
    nb_correct_robust = [nb_correct_robust_pred] + nb_correct_robust

    return (eps_range, nb_correct_original, nb_correct_robust)

def plot_graph(eps_range, nb_correct_original,nb_correct_robust,title):
    fig, ax = plt.subplots()
    ax.plot(np.array(eps_range), np.array(nb_correct_original), 'b--', label='Original classifier')
    ax.plot(np.array(eps_range), np.array(nb_correct_robust), 'r--', label='Robust classifier')

    legend = ax.legend(loc='upper center', shadow=True, fontsize='large')
    legend.get_frame().set_facecolor('#00FFCC')
    plt.title(title)
    plt.xlabel('Attack strength (eps)')
    plt.ylabel('Correct predictions')
    plt.savefig('Pixelevaluation{}.jpg'.format(title))


def create_img(x,filename):
    plt.imsave(filename,x[5,...].squeeze() )


def create_report(nb_correct_pred,nb_correct_adv_pred,nb_correct_robust_pred,nb_correct_adv_robust_pred,graph_name,dataset):
    htmlfile = open('report.html','w')
    html = f"""
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Robustness Analyzer</title>
    <link rel="stylesheet" href="https://bootswatch.com/5/zephyr/bootstrap.min.css">
     <script src=
"https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.9.2/html2pdf.bundle.js">
    </script>
</head>
<body>
    <div class="container" id="makepdf">
    <br>
    <center><h3>Robustness Analysis Report</h3></center>
    <br>
    <button class ="btn btn-primary" id="button">Generate PDF</button>
    <br><br>
    <p>Attack : Pixel Attack </p>
    <p>Defense : Adversarial Training </p>
    <p>Model : {dataset} </p>
    <br><br>
    <p>Accuracy of the original model with clean dataset (100) : {nb_correct_pred} %</p>
    <p>Number of correct predictions : {nb_correct_pred} </p>
    <p>Number of incorrect predictions : {100-nb_correct_pred} </p> <br><br>
    <p>Accuracy of the original model with adversarial data generated (100) : {nb_correct_adv_pred} %</p>
    <p>Number of correct predictions : {nb_correct_adv_pred} </p>
    <p>Number of incorrect predictions : {100-nb_correct_adv_pred} </p> <br><br>
    <p>Accuracy of the robust model with clean dataset trained with attacks (100) : {nb_correct_robust_pred} %</p>
    <p>Number of correct predictions : {nb_correct_robust_pred} </p>
    <p>Number of incorrect predictions : {100-nb_correct_robust_pred} </p> <br><br>
    <p>Accuracy of the robust model with adversarial data generated (100) after defense : {nb_correct_adv_robust_pred} % </p>
    <p>Number of correct predictions : {nb_correct_adv_robust_pred} </p>
    <p>Number of incorrect predictions : {100-nb_correct_adv_robust_pred} </p> <br><br>
    <hr>
    <center><h3>Graphical Representation of Robustness and Attacks applied</h3>
    <img src="./{graph_name}" alt="Graph for attack strength vs correct predictions">
   <div class="row">
        <div class="col-md-4">
            <p>Original image</p>
            <img src="./original_image.jpg" alt="original image" height="200px" width="200px">
        </div>
        <div class="col-md-4">
            <p>Image after attack</p>
            <img src="./attacked_image.jpg" alt="image under attack"  height="200px" width="200px">
        </div>
        <div class="col-md-4">
            <p>Image after defense</p>
            <img src="./robust_image.jpg" alt="image after defense"  height="200px" width="200px">
        </div>
    </div>
    <br><br><br><br>
    </center>
    </div>
    
</body>
<script>
    var button = document.getElementById('button');
    var makepdf = document.getElementById('makepdf');
    button.addEventListener('click', function () 
    {{
        
        window.print();
    
    }});
</script>
</html>
    """
    htmlfile.write(html)
    htmlfile.close()



def Pixel_main(dataset):
    dataset_name = {'mnist':'MNIST - Handwritten number classification model','cifar10':'Cifar10 - Random image classification model'}
    x_train, y_train, x_test, y_test, min_, max_ = load_data(dataset)
    classifier = create_model(min_,max_,dataset)
    x_test_pred , nb_correct_pred = original_accuracy(classifier,x_test,y_test)
    x_test_adv, attacker = attack(classifier,x_test)
    x_test_adv_pred, nb_correct_adv_pred = attacked_accuracy(classifier,x_test,y_test)
    robust_classifier = create_robust_model(min_,max_,dataset)
    attacker_robust = attack_robust(robust_classifier,x_train, y_train)
    nb_correct_robust_pred = robust_with_attack(robust_classifier,x_test,y_test)
    nb_correct_adv_robust_pred,x_test_adv_robust = robust_accuracy(robust_classifier,x_test,y_test)
    eps_range, nb_correct_original, nb_correct_robust = evaluate(attacker,attacker_robust,classifier,robust_classifier,nb_correct_pred,nb_correct_robust_pred,x_test,y_test)
    # fgsm.plot_graph(eps_range,x_test[:15],nb_correct_pred,'Before Attack') # Before Attack graph
    # fgsm.plot_graph(eps_range,nb_correct_original,nb_correct_adv_pred, 'After Attack')  #Before defense graph
    graph_name = plot_graph(eps_range, nb_correct_original,nb_correct_robust,'After Defense') #after defense
    create_img(x_test,'original_image.jpg')
    create_img(x_test_adv,'attacked_image.jpg')
    create_img(x_test_adv_robust,'robust_image.jpg')
    create_report(nb_correct_pred,nb_correct_adv_pred,nb_correct_robust_pred,nb_correct_adv_robust_pred,graph_name,dataset_name[dataset])

