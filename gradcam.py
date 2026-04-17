import numpy as np
import tensorflow as tf
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

CLASS_NAMES = [
    "Albrecht_Durer",        
    "Boris_Kustodiev",       
    "Camille_Pissarro",      
    "Childe_Hassam",         
    "Claude_Monet",          
    "Edgar_Degas",           
    "Eugene_Boudin",         
    "Gustave_Dore",          
    "Ilya_Repin",            
    "Ivan_Aivazovsky",      
    "Ivan_Shishkin",         
    "John_Singer_Sargent",   
    "Marc_Chagall",          
    "Martiros_Saryan",       
    "Nicholas_Roerich",      
    "Pablo_Picasso",         
    "Paul_Cezanne",          
    "Pierre_Auguste_Renoir", 
    "Pyotr_Konchalovsky",    
    "Raphael_Kirchner",      
    "Rembrandt",             
    "Salvador_Dali",         
    "Vincent_van_Gogh",      
]




def _load_image(img_path):
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    arr = keras.utils.img_to_array(img) / 255.0
    return tf.cast(tf.expand_dims(arr, 0), tf.float32)


def _find_gradcam_layer(model, model_type):
    mt = model_type.lower()
    if mt == "improved_cnn":
        name = None
        for layer in model.layers:
            if "activation" in layer.name:
                name = layer.name
        return name
    if mt == "vgg16":
        return "block5_conv3"
    if mt in ("resnet", "resnet_ft"):
        return "conv5_block3_out"
    name = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            name = layer.name
    return name


def _find_layer_obj(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                if sub.name == layer_name:
                    return sub
    raise ValueError(f"Layer '{layer_name}' not found.")


#Creo GradCAM 

def _make_gradcam_heatmap(img_tensor, model, model_type):
    
    layer_name = _find_gradcam_layer(model, model_type)

    if model_type.lower() == "improved_cnn":
        # Sequential: iterate layer by layer, skip data augmentation
        with tf.GradientTape() as tape:
            x = img_tensor
            last_conv_output = None
            for layer in model.layers:
                if "sequential" in layer.name:
                    continue
                x = layer(x, training=False)
                if layer.name == layer_name:
                    last_conv_output = x
                    tape.watch(last_conv_output)
            preds = x
            pred_idx = int(tf.argmax(preds[0]))
            class_channel = preds[:, pred_idx]

    else:
        # Functional models: 2-output sub-model e
        target_layer = _find_layer_obj(model, layer_name)
        grad_model = keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_output, preds = grad_model(img_tensor, training=False)
            tape.watch(last_conv_output)
            pred_idx = int(tf.argmax(preds[0]))
            class_channel = preds[:, pred_idx]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy(), pred_idx, preds[0].numpy()


def _overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_colors = mpl.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * alpha + img
    return keras.utils.array_to_img(superimposed)


def make_gradcam(img_path, model, model_type="improved_cnn", alpha=0.4):

    img_tensor = _load_image(img_path)
    heatmap, pred_idx, probs = _make_gradcam_heatmap(img_tensor, model, model_type)
    overlay = _overlay_heatmap(img_path, heatmap, alpha)

    pred_name  = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    print(f"Predicted: {pred_name.replace('_', ' ')}  ({confidence*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        f"[{model_type}]  ->  {pred_name.replace('_', ' ')}  ({confidence*100:.1f}%)",
        fontsize=11, fontweight="bold"
    )
    axes[0].imshow(keras.utils.load_img(img_path))
    axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def compare_best_worst_classes(
    y_true,
    y_pred,
    test_df,
    model,
    model_type="improved_cnn",
    alpha=0.4,
    n_images=3,
):
    """
 finds the worst and the best through y_true/y_pred 
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    f1_per_class = {int(k): v["f1-score"] for k, v in report.items() if k.isdigit()}

    best_idx  = max(f1_per_class, key=f1_per_class.get)
    worst_idx = min(f1_per_class, key=f1_per_class.get)
    best_name  = CLASS_NAMES[best_idx]
    worst_name = CLASS_NAMES[worst_idx]
    best_f1    = f1_per_class[best_idx]
    worst_f1   = f1_per_class[worst_idx]

    print(f"  Best  : {best_name.replace('_', ' ')}  (f1 = {best_f1:.2f})")
    print(f"  Worst : {worst_name.replace('_', ' ')}  (f1 = {worst_f1:.2f})")

    groups = {
        f"Best - {best_name.replace('_', ' ')} (f1={best_f1:.2f})":
            test_df[test_df["label"] == best_idx]["image_path"].head(n_images).tolist(),
        f"Worst - {worst_name.replace('_', ' ')} (f1={worst_f1:.2f})":
            test_df[test_df["label"] == worst_idx]["image_path"].head(n_images).tolist(),
    }

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, n_images, figsize=(4 * n_images, 4 * n_groups))
    axes = np.array(axes)
    if axes.ndim == 1:
        axes = axes.reshape(n_groups, n_images)

    for g_idx, (group_label, img_paths) in enumerate(groups.items()):
        for col, img_path in enumerate(img_paths):
            img_tensor = _load_image(img_path)
            heatmap, _, _ = _make_gradcam_heatmap(img_tensor, model, model_type)
            overlay = _overlay_heatmap(img_path, heatmap, alpha)

            axes[g_idx, col].imshow(overlay)
            axes[g_idx, col].axis("off")
            if col == 0:
                axes[g_idx, col].set_title(group_label, fontsize=11,
                                            fontweight="bold", loc="left", pad=8)

    fig.suptitle(f"Grad-CAM  |  model: {model_type}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
