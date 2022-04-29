from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from cancernet.cancernet import CancerNet
from cancernet import config
from imutils import paths
import numpy as np
import os
import matplotlib.pylab as plt

#setting paths for testing,validation and testing
training_path=list(paths.list_images(config.TRAIN_PATH)) 
len_training_path=len(training_path)
validating_path=list(paths.list_images(config.VAL_PATH))
len_validating_path=len(validating_path)
testing_path=list(paths.list_images(config.TEST_PATH))
len_testing_path=len(testing_path)




#class weight calculation

training_labels=[int(p.split(os.path.sep)[-2]) for p in training_path]
training_labels=np_utils.to_categorical(training_labels)

class_total=training_labels.sum(axis=0)
class_weight=class_total.max()/class_total
classWeightDict={0: class_weight[0], 1:class_weight[1]}


#data augmentation step
training_augmentation=ImageDataGenerator(rotation_range=20,width_shift_range=0.1, height_shift_range=0.1,shear_range=0.05,zoom_range=0.05,
                                         fill_mode="nearest",horizontal_flip=True,vertical_flip=True,rescale=1/255.0)

validating_augmentaion=ImageDataGenerator(rescale=1/255.0)

# no of epochs , learning_rate and batch size declaration
no_of_epochs=40

intial_learning_rate=0.0001

BS=32 #batch size

#generating batches of augmented data using flow_from_directory method of ImageDataGenerator
training_generator=training_augmentation.flow_from_directory(
    config.TRAIN_PATH,
    target_size=(48, 48),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=True
)

validating_generator=validating_augmentaion.flow_from_directory(
    config.VAL_PATH,
    target_size=(48, 48),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

testing_generator=validating_augmentaion.flow_from_directory(
    config.TEST_PATH,
    target_size=(48, 48),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=32,
    shuffle=False
)

#model build and compile using Adagrad optimizer
model=CancerNet.build(width=48,height=48,depth=3,classes=2);
opt=Adagrad(learning_rate=intial_learning_rate,decay=intial_learning_rate/no_of_epochs)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

#model fit
history=model.fit(training_generator,
            epochs=no_of_epochs,
            validation_data=validating_generator,
            class_weight=classWeightDict,
            steps_per_epoch=len_training_path//BS,
            validation_steps=len_validating_path//BS)


print ("Evaluation of the model:")
testing_generator.reset();

#Generates predictions for the input samples from a data generator.
predicted_classes=model.predict_generator(testing_generator,steps=(len_testing_path//BS)+1)
predicted_classes=np.argmax(predicted_classes,axis=1)
print(classification_report(testing_generator.classes, predicted_classes, target_names=testing_generator.class_indices.keys()))

# drawing confusion matrix
cm=confusion_matrix(testing_generator.classes,predicted_classes)

total_no_of_predictions_made=sum(sum(cm))

no_of_correct_predictions=cm[0,0]+cm[1,1]

# classification accuracy method to calculate accuracy
classification_accuracy=no_of_correct_predictions/total_no_of_predictions_made

specificity=cm[1,1]/(cm[1,0]+cm[1,1])

sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])

print(cm)
print(f'Accuracy: {classification_accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')



plt.title('Model Accuracy using Adagrad Optimizer')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.savefig('plot1.png')

#model build and compile using RMSprop optimizer
model=CancerNet.build(width=48,height=48,depth=3,classes=2);


opt=RMSprop(learning_rate=intial_learning_rate)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

#model fit
history=model.fit(training_generator,
            epochs=no_of_epochs,
            validation_data=validating_generator,
            class_weight=classWeightDict,
            steps_per_epoch=len_training_path//BS,
            validation_steps=len_validating_path//BS)


print ("Evaluation of the model:")
testing_generator.reset();

#Generates predictions for the input samples from a data generator.
predicted_classes=model.predict_generator(testing_generator,steps=(len_testing_path//BS)+1)
predicted_classes=np.argmax(predicted_classes,axis=1)
print(classification_report(testing_generator.classes, predicted_classes, target_names=testing_generator.class_indices.keys()))

# drawing confusion matrix
cm=confusion_matrix(testing_generator.classes,predicted_classes)

total_no_of_predictions_made=sum(sum(cm))

no_of_correct_predictions=cm[0,0]+cm[1,1]

# classification accuracy method to calculate accuracy
classification_accuracy=no_of_correct_predictions/total_no_of_predictions_made

specificity=cm[1,1]/(cm[1,0]+cm[1,1])

sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])

print(cm)
print(f'Accuracy: {classification_accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')


plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Model Accuracy using RMSprop Optimizer')
plt.savefig('plot2.png')


#model build and compile using Adam optimizer
model=CancerNet.build(width=48,height=48,depth=3,classes=2);


opt=Adam(learning_rate=intial_learning_rate)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

#model fit
history=model.fit(training_generator,
            epochs=no_of_epochs,
            validation_data=validating_generator,
            class_weight=classWeightDict,
            steps_per_epoch=len_training_path//BS,
            validation_steps=len_validating_path//BS)


print ("Evaluation of the model:")
testing_generator.reset();

#Generates predictions for the input samples from a data generator.
predicted_classes=model.predict_generator(testing_generator,steps=(len_testing_path//BS)+1)
predicted_classes=np.argmax(predicted_classes,axis=1)
print(classification_report(testing_generator.classes, predicted_classes, target_names=testing_generator.class_indices.keys()))

# drawing confusion matrix
cm=confusion_matrix(testing_generator.classes,predicted_classes)

total_no_of_predictions_made=sum(sum(cm))

no_of_correct_predictions=cm[0,0]+cm[1,1]

# classification accuracy method to calculate accuracy
classification_accuracy=no_of_correct_predictions/total_no_of_predictions_made

specificity=cm[1,1]/(cm[1,0]+cm[1,1])

sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])

print(cm)
print(f'Accuracy: {classification_accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')



plt.title('Model Accuracy using Adam Optimizer')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.savefig('plot3.png')