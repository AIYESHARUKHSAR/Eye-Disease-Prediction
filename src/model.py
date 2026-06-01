import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_model(num_classes=config.NUM_CLASSES,
                input_shape=config.INPUT_SHAPE,
                dropout_rate1=0.5,
                dropout_rate2=0.4):
    """Build EfficientNetB4 multi-label classification model.

    Phase 1: backbone frozen, only custom head is trained.
    Phase 2: top backbone layers unfrozen for fine-tuning.
    """
    backbone = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    backbone.trainable = False  # frozen for Phase 1

    inputs = tf.keras.Input(shape=input_shape, name='fundus_input')
    x = backbone(inputs, training=False)

    x = layers.GlobalAveragePooling2D(name='gap')(x)

    x = layers.Dense(512, name='dense_512')(x)
    x = layers.BatchNormalization(name='bn_512')(x)
    x = layers.Activation('relu', name='relu_512')(x)
    x = layers.Dropout(dropout_rate1, name='dropout_512')(x)

    x = layers.Dense(256, name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.Activation('relu', name='relu_256')(x)
    x = layers.Dropout(dropout_rate2, name='dropout_256')(x)

    # Sigmoid output — independent probability per disease class
    outputs = layers.Dense(num_classes, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='EfficientNetB4_ODIR')
    return model, backbone


def compile_model(model, learning_rate=config.PHASE1_LR):
    """Compile with binary cross-entropy for multi-label classification."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model


def unfreeze_top_layers(backbone, num_layers=config.FINE_TUNE_LAYERS):
    """Unfreeze the top `num_layers` of the backbone for Phase 2 fine-tuning."""
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False

    trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"Fine-tuning: {trainable}/{len(backbone.layers)} backbone layers trainable")
    return backbone


def get_callbacks(checkpoint_path=config.CHECKPOINT_PATH):
    """Return standard training callbacks."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0),
        tf.keras.callbacks.CSVLogger('training_log.csv', append=True),
    ]


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def model_summary(model):
    model.summary()
    total = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"\nTotal:       {total:,}")
    print(f"Trainable:   {trainable:,}")
    print(f"Frozen:      {total - trainable:,}")
