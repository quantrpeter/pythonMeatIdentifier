"""
Meat Identification Model
Uses transfer learning with MobileNetV2 for efficient inference
Identifies raw meat in various packaging types (vacuum, aluminum foil, frozen)
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


class MeatIdentifier:
    """
    Deep learning model for identifying raw meat in images.
    Supports detection of meat packaged in vacuum, aluminum foil, or frozen state.
    """
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=4):
        """
        Initialize the meat identifier model.
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of classes (default: 4)
                - 0: No meat
                - 1: Vacuum packaged meat
                - 2: Aluminum foil packaged meat
                - 3: Frozen meat
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = [
            'No Meat',
            'Vacuum Packaged Meat',
            'Aluminum Foil Packaged Meat',
            'Frozen Meat'
        ]
        
    def build_model(self, trainable_layers=20):
        """
        Build the model using transfer learning with MobileNetV2.
        
        Args:
            trainable_layers: Number of top layers to unfreeze for training
        """
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model initially
        base_model.trainable = True
        
        # Unfreeze the top layers
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
    def train(self, train_data_dir, validation_data_dir, epochs=20, batch_size=32):
        """
        Train the model on a dataset.
        
        Args:
            train_data_dir: Path to training data directory
            validation_data_dir: Path to validation data directory
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            validation_data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_meat_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, image):
        """
        Make a prediction on a single image.
        
        Args:
            image: Preprocessed image array (normalized, correct shape)
            
        Returns:
            Dictionary with class name and confidence
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return {
            'class': self.class_names[class_idx],
            'class_index': int(class_idx),
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(predictions[0][i]) 
                for i in range(len(self.class_names))
            }
        }
    
    def save_model(self, filepath='models/meat_identifier.h5'):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/meat_identifier.h5'):
        """Load a pre-trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise ValueError("Model not built.")
        return self.model.summary()
