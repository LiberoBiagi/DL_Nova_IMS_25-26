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
from keras.layers import Input, Normalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Resizing, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# model training and evaluation
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, AUC, F1Score
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# pre-trained models
import keras_hub
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess_input

#------------------------------------------------------------------------------------------------------------------------------
#                                                    MODEL BUILDING FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------------

def Our_Net(input_shape=(224, 224, 3),
             num_classes=23,
             data_augmentation=None):
    
    model = Sequential([
        Input(shape=input_shape),
        Resizing(224, 224),
        Normalization(),  # normalize pixel values to [0, 1]
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

    return model


def ResNet50___(input_shape=(224, 224, 3),
            num_classes=23,
            data_augmentation=None):

    # base model with pre-trained weights on ImageNet
    resnet_base = ResNet50(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
    )

    for layer in resnet_base.layers:
        layer.trainable = False

    inputs_resnet = layers.Input(shape=input_shape)

    if data_augmentation is not None:
        x = data_augmentation(inputs_resnet)
    else:
        x = inputs_resnet
    x = layers.Resizing(224, 224)(x)
    x = resnet_preprocess_input(x)
    x = resnet_base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs_resnet, x)

    return model


def  InceptionV3__(input_shape=(224, 224, 3),  #### CHECK SIZE
                  num_classes=23,
                  data_augmentation=None):
    
    # base model with pre-trained weights on ImageNet
    inceptionv3_base = InceptionV3(weights='imagenet',
                                include_top=False, # exclude the fully connected layers at the top of the network
                                input_shape=input_shape)

    for layer in inceptionv3_base.layers:
        layer.trainable = False

    inputs_inceptionv3 = layers.Input(shape=input_shape)

    if data_augmentation is not None:
        x = data_augmentation(inputs_inceptionv3)
    else:
        x = inputs_inceptionv3
    x = layers.Resizing(299, 299)(x)
    x = inception_preprocess_input(x)
    x = inceptionv3_base(x, training=False) 
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs_inceptionv3, x)

    return model


def ViT__(input_shape=(224, 224, 3), 
          num_classes=23,
          data_augmentation=None):
    
    vit_model = keras_hub.models.ViTImageClassifier.from_preset(
        "vit_base_patch16_224_imagenet",
        num_classes=num_classes,
        activation='softmax',
        pooling='gap',
        dropout=0.3
    )
    
    for layer in vit_model.backbone.layers:
        layer.trainable = False  

    inputs_vit = layers.Input(shape=input_shape)
    
    x = inputs_vit
    if data_augmentation is not None:
        x = data_augmentation(x)
    
    x = layers.Resizing(224, 224)(x)
    outputs_vit = vit_model(x)    
    model = Model(inputs_vit, outputs_vit)
    return model

#------------------------------------------------------------------------------------------------------------------------------
#                                                    MODEL EVALUATION FUNCTIONS
#------------------------------------------------------------------------------------------------------------------------------

def complete_classification_report (y_true, y_pred, model_name):

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print(f"Model: {model_name}")
    print(f"F1 Macro:    {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")


    print(f"\nClassification Report {model_name}:")
    print(classification_report(y_true, y_pred))

    return f1_macro, f1_weighted

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
