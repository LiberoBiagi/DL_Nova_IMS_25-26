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
    arr = keras.utils.img_to_array(img)  
    return tf.cast(tf.expand_dims(arr, 0), tf.float32)


def _make_gradcam_heatmap(img_tensor, model):
    """
    Grad-CAM for the final fine-tuned ResNet50 model.

    Model structure (from summary):
        input_layer_1 -> sequential -> resizing -> get_item* -> stack -> add
        -> resnet50 -> global_average_pooling -> dense* -> dense_5 (output)

    We build a two-output Keras model using the symbolic tensors already
    in the graph: model.input -> [conv5_block3_out, model.output].
    This is the only approach that keeps the gradient path intact.
    """
    # locate resnet50 sublayer and its conv5_block3_out layer
    resnet_sub = None
    for layer in model.layers:
        if "resnet50" in layer.name.lower():
            resnet_sub = layer
            break
    if resnet_sub is None:
        raise ValueError("Could not find resnet50 sublayer. Check model.summary().")


    resnet_grad_model = keras.models.Model(
        inputs=resnet_sub.inputs,
        outputs=resnet_sub.outputs
    )

    # run the outer model up to resnet50 to get its input tensor
    # layers before resnet50: sequential, resizing, get_item x3, stack, add
    pre_layers = []
    for layer in model.layers:
        if layer.name == resnet_sub.name:
            break
        if layer.__class__.__name__ == "InputLayer":
            continue
        pre_layers.append(layer)

    with tf.GradientTape() as tape:
        x = img_tensor
        for layer in pre_layers:
            x = layer(x, training=False)
        # x is now the input to resnet50
        conv_out = resnet_grad_model(x, training=False)
        tape.watch(conv_out)
        # continue through post-resnet layers
        post_x = conv_out
        found = False
        for layer in model.layers:
            if found:
                post_x = layer(post_x, training=False)
            if layer.name == resnet_sub.name:
                found = True
        preds = post_x
        pred_idx = int(tf.argmax(preds[0]))
        class_score = preds[:, pred_idx]

    grads = tape.gradient(class_score, conv_out)

    if grads is None:
        raise ValueError("Gradients are None. Check model structure.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_idx, preds[0].numpy()


def _overlay_heatmap(img_path, heatmap, alpha=0.4, colormap="jet"):
    """
    Superimpose a coloured Grad-CAM heatmap over the original image.

    Parameters
    ----------
    alpha     : blending weight for the heatmap (higher = more vivid).
    colormap  : any matplotlib colormap name.
                'jet'      – classic blue→red (default).
                'inferno'  – dark purple→yellow (more intense/dramatic).
                'turbo'    – perceptually-uniform, very vivid.
    """
    img = keras.utils.img_to_array(keras.utils.load_img(img_path))

    heatmap_uint8 = np.uint8(255 * heatmap)
    cmap_colors = mpl.colormaps[colormap](np.arange(256))[:, :3]
    colored = cmap_colors[heatmap_uint8]
    colored_img = keras.utils.array_to_img(colored)
    colored_img = colored_img.resize((img.shape[1], img.shape[0]))
    colored_arr = keras.utils.img_to_array(colored_img)

    superimposed = colored_arr * alpha + img
    return keras.utils.array_to_img(superimposed)


def make_gradcam(img_path, model, model_type=None, alpha=0.4, colormap="jet"):
    """
    Run Grad-CAM on a single image and display original + heatmap side-by-side.

    Parameters
    ----------
    img_path  : path to the image file.
    model     : the final Keras model (best_deep_resnet_ft2_model).
    model_type: ignored (kept for API compatibility with older code).
    alpha     : heatmap blending strength.
    colormap  : matplotlib colormap for the heatmap overlay.
    """
    img_tensor = _load_image(img_path)
    heatmap, pred_idx, probs = _make_gradcam_heatmap(img_tensor, model)
    overlay = _overlay_heatmap(img_path, heatmap, alpha=alpha, colormap=colormap)

    pred_name = CLASS_NAMES[pred_idx].replace("_", " ")
    confidence = probs[pred_idx]
    print(f"Predicted: {pred_name}  ({confidence * 100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        f"Grad-CAM  →  {pred_name}  ({confidence * 100:.1f}%)",
        fontsize=11, fontweight="bold",
    )
    axes[0].imshow(keras.utils.load_img(img_path))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def compare_best_worst_classes(
    y_true,
    y_pred,
    test_df,
    model,
    model_type=None,
    alpha_best=0.65,   
    alpha_worst=0.4,   
    colormap_best="turbo",   
    colormap_worst="jet",    
    n_images=3,
):
    """
    Find the best and worst predicted classes (by F1-score), then show
    Grad-CAM overlays for each — two separate figures, one per class.

    Best class  : higher alpha + 'turbo' colourmap for more intense colours.
    Worst class : standard alpha + 'jet' colourmap.

    Parameters
    ----------
    y_true / y_pred   : integer label arrays from the test set.
    test_df           : DataFrame with columns 'label' and 'image_path'.
    model             : the final Keras model.
    alpha_best        : heatmap intensity for the best-class figure (0–1).
    alpha_worst       : heatmap intensity for the worst-class figure (0–1).
    colormap_best     : matplotlib colormap for the best-class heatmaps.
    colormap_worst    : matplotlib colormap for the worst-class heatmaps.
    n_images          : number of sample images to show per class.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    f1_per_class = {int(k): v["f1-score"] for k, v in report.items() if k.isdigit()}

    best_idx  = max(f1_per_class, key=f1_per_class.get)
    worst_idx = min(f1_per_class, key=f1_per_class.get)
    best_name  = CLASS_NAMES[best_idx].replace("_", " ")
    worst_name = CLASS_NAMES[worst_idx].replace("_", " ")
    best_f1    = f1_per_class[best_idx]
    worst_f1   = f1_per_class[worst_idx]

    print(f"  Best  : {best_name}  (F1 = {best_f1:.2f})")
    print(f"  Worst : {worst_name}  (F1 = {worst_f1:.2f})")

    groups = [
        {
            "title":    f"BEST CLASS — {best_name}  (F1 = {best_f1:.2f})",
            "color":    "green",
            "class_idx": best_idx,
            "alpha":    alpha_best,
            "colormap": colormap_best,
        },
        {
            "title":    f"WORST CLASS — {worst_name}  (F1 = {worst_f1:.2f})",
            "color":    "red",
            "class_idx": worst_idx,
            "alpha":    alpha_worst,
            "colormap": colormap_worst,
        },
    ]

    for group in groups:
        img_paths = (
            test_df[test_df["label"] == group["class_idx"]]["image_path"]
            .head(n_images)
            .tolist()
        )

        fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8))
        axes = np.array(axes).reshape(2, n_images)

        fig.suptitle(
            group["title"],
            fontsize=13, fontweight="bold", color=group["color"],
        )

        for col, img_path in enumerate(img_paths):
            img_tensor = _load_image(img_path)
            heatmap, pred_idx, probs = _make_gradcam_heatmap(img_tensor, model)
            confidence = probs[pred_idx] * 100
            pred_name  = CLASS_NAMES[pred_idx].replace("_", " ")
            overlay    = _overlay_heatmap(
                img_path, heatmap,
                alpha=group["alpha"],
                colormap=group["colormap"],
            )

            # extract painting name from file path, strip extension, fix underscores
            import os
            painting_name = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ")
            # truncate if very long so it fits under the image
            if len(painting_name) > 30:
                painting_name = painting_name[:28] + "…"

            # row 0: original image + painting name as title
            axes[0, col].imshow(keras.utils.load_img(img_path))
            axes[0, col].set_title(painting_name, fontsize=8, style="italic")
            axes[0, col].axis("off")
            if col == 0:
                axes[0, col].set_ylabel("Original", fontsize=11, fontweight="bold")

            # row 1: Grad-CAM overlay + prediction + confidence
            axes[1, col].imshow(overlay)
            axes[1, col].set_title(
                f"Pred: {pred_name}\n{confidence:.1f}%",
                fontsize=9,
            )
            axes[1, col].axis("off")
            if col == 0:
                axes[1, col].set_ylabel("Grad-CAM", fontsize=11, fontweight="bold")

        plt.tight_layout()
        plt.show()
