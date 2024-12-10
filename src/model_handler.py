# path: ~/Develop/kamp/src/model_handler.py
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from visual import Visualizer

class ModelHandler:
    def __init__(self, model_dir='models', figure_dir='figures' ,viz=None):
        self.model_dir = model_dir
        self.figure_dir = figure_dir
        self.visualizer = viz if viz is not None else Visualizer()
        os.makedirs(model_dir, exist_ok=True)
        
    def create_model(self, input_dim):
        """DNN 모델 생성"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_kfold(self, X_train, y_train, X_test, y_test, n_splits=3):
        """K-Fold 교차검증을 통한 모델 학습"""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"\nTraining Fold {fold + 1}")
            
            # 폴드별 데이터 분할
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # 모델 생성 및 학습
            model = self.create_model(X_train.shape[1])
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=3
                )
            ]
            
            history = model.fit(
                X_fold_train, y_fold_train,
                epochs=100,
                validation_data=(X_fold_val, y_fold_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # 테스트 세트에 대한 평가
            predictions = model.predict(X_test)
            pred_binary = predictions > 0.5
            
            # 성능 메트릭스 계산
            metrics = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test, pred_binary),
                'f1': f1_score(y_test, pred_binary),
                'confusion_matrix': confusion_matrix(y_test, pred_binary),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'final_train_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1])
            }
            fold_metrics.append(metrics)
            
            # 모델 저장
            model.save(os.path.join(self.model_dir, f'model_fold_{fold+1}.keras'))
            
            # 학습 곡선 시각화
            self.visualizer.plot_learning_curves(
                history, 
                fold=fold+1,
                save_path=os.path.join(self.figure_dir, f'learning_curves_fold_{fold+1}.png')
            )
            
            # Confusion Matrix 시각화
            self.visualizer.plot_confusion_matrix(
                y_test, 
                pred_binary,
                fold=fold+1,
                save_path=os.path.join(self.figure_dir, f'confusion_matrix_fold_{fold+1}.png')
            )
        
        return fold_metrics