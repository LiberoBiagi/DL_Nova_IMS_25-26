#------------------------------------------------------------------------------------------------------------------------------
#                                                            IMPORTS
#------------------------------------------------------------------------------------------------------------------------------

# visualization
import matplotlib.pyplot as plt

# model building
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras import Model, layers
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Resizing, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# model training and evaluation
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, AUC, F1Score
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# pre-trained models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

#------------------------------------------------------------------------------------------------------------------------------
#                                                    MODEL BUILDING FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------------

def Our_Net(input_shape=(224, 224, 3), num_classes=23):
    model = Sequential([
        Input(shape=input_shape),
        Resizing(224, 224),
        data_augmentation,

        # Block 1
        Conv2D(32, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2), padding='same'),  # → 112x112

        # Block 2
        Conv2D(64, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2), padding='same'),  # → 56x56

        # Block 3
        Conv2D(128, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2), padding='same'),  # → 28x28

        # Blocco 4
        Conv2D(256, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2,2), padding='same'),  # → 14x14

        # Block 5
        Conv2D(256, (3,3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling2D(),

        # Classifier
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def ResNet50(input_shape=(224, 224, 3), num_classes=23):
    ## Completar


def  InceptionV3(input_shape=(224, 224, 3), num_classes=23):
    ## Completar

def ViT(input_shape=(224, 224, 3), num_classes=23):
    ## Completar    

#------------------------------------------------------------------------------------------------------------------------------
#                                                    MODEL EVALUATION FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------------

def classificattion_report (y_true, y_pred):

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")


    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def accuracy_loss_curves(history_dict, model_name):
     
    # get training and validation accuracy and loss from history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']

    epochs = range(1, len(loss_values) + 1) # len(loss_values) = len(acc_values) = number of epochs

    # create figure
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'{model_name} - Training and Validation Curves', fontsize=16)

    # loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()