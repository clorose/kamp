import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from scipy import stats

class Visualizer:
    def __init__(self):
        # 시각화 스타일 설정
        plt.style.use('seaborn-v0_8')
        
    def plot_data_distribution(self, df, save_path=None):
        """데이터 분포 시각화"""
        plt.figure(figsize=(15, 10))
        
        # 클래스 분포
        plt.subplot(2, 2, 1)
        sns.countplot(data=df, x='passorfail')
        plt.title('Target Class Distribution')
        
        # 주요 변수들의 상관관계 히트맵
        plt.subplot(2, 2, 2)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlations')
        
        # SpindleLoad_max 분포 (양품/불량품 구분)
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x='passorfail', y='SpindleLoad_max')
        plt.title('SpindleLoad_max Distribution by Class')
        
        # SpindleLoad vs ServoCurrent 산점도
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x='SpindleLoad_max', y='ServoCurrent_X_mean', 
                       hue='passorfail', alpha=0.6)
        plt.title('SpindleLoad vs ServoCurrent')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_preprocessing_comparison(self, original_data, processed_data, save_path=None):
        """전처리 전후 비교 시각화"""
        plt.figure(figsize=(15, 5))
        
        # 원본 데이터 분포
        plt.subplot(1, 2, 1)
        sns.countplot(data=original_data, x='passorfail')
        plt.title('Class Distribution (Original)')
        
        # SMOTE 적용 후 데이터 분포
        plt.subplot(1, 2, 2)
        sns.countplot(data=processed_data, x='passorfail')
        plt.title('Class Distribution (After SMOTE)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_learning_curves(self, history, fold=None, save_path=None):
        """학습 곡선 시각화"""
        plt.figure(figsize=(12, 4))
        
        # Loss 곡선
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        title = 'Model Loss'
        if fold is not None:
            title += f' (Fold {fold})'
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy 곡선
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        title = 'Model Accuracy'
        if fold is not None:
            title += f' (Fold {fold})'
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, fold=None, save_path=None):
        """Confusion Matrix 시각화"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        
        title = 'Confusion Matrix'
        if fold is not None:
            title += f' (Fold {fold})'
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_feature_importance(self, df, target_col='passorfail', save_path=None):
        """특성 중요도 시각화 (T-test 기반)"""
        plt.figure(figsize=(12, 6))
        
        # 각 특성별 T-test 수행
        feature_importance = {}
        for col in df.columns:
            if col != target_col and df[col].dtype in ['int64', 'float64']:
                t_stat, p_val = stats.ttest_ind(
                    df[df[target_col]==1][col],
                    df[df[target_col]==0][col],
                    equal_var=False
                )
                feature_importance[col] = -np.log10(p_val)  # p-value를 로그 스케일로 변환
        
        # 중요도 순으로 정렬하여 시각화
        importance_df = pd.DataFrame({'feature': feature_importance.keys(),
                                    'importance': feature_importance.values()})
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance (-log10(p-value))')
        plt.xlabel('-log10(p-value)')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_fold_comparison(self, fold_metrics, save_path=None):
        """K-Fold 교차검증 결과 비교 시각화"""
        plt.figure(figsize=(10, 5))
        
        # 성능 지표 추출
        folds = [m['fold'] for m in fold_metrics]
        accuracies = [m['accuracy'] for m in fold_metrics]
        f1_scores = [m['f1'] for m in fold_metrics]
        
        x = np.arange(len(folds))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy')
        plt.bar(x + width/2, f1_scores, width, label='F1 Score')
        
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title('Model Performance Across Folds')
        plt.xticks(x, [f'Fold {f}' for f in folds])
        plt.legend()
        
        # 평균 성능 표시
        plt.axhline(y=np.mean(accuracies), color='b', linestyle='--', alpha=0.3)
        plt.axhline(y=np.mean(f1_scores), color='orange', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_initial_distributions(self, df, save_path=None):
        """변수별 초기 분포 시각화 (가이드북 스타일)"""
        cols = [col for col in df.columns if (col != 'SerialNo') & 
            (col != 'ReceivedDateTime') & (col != 'passorfail')]

        n_cols = len(cols)
        n_rows = (n_cols-1) // 3 + 1

        plt.figure(figsize=(20, 6*n_rows))

        for i, col in enumerate(cols, 1):
            plt.subplot(n_rows, 3, i)
            plt.title(col)
            plt.hist(df[col])
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_class_distributions(self, df, save_path=None):
        """양품/불량별 분포 시각화 (가이드북 스타일)"""
        cols = [col for col in df.columns if (col != 'SerialNo') & 
            (col != 'ReceivedDateTime') & (col != 'passorfail')]

        n_cols = len(cols)
        n_rows = (n_cols-1) // 3 + 1

        plt.figure(figsize=(20, 6*n_rows))
        pos_df = df[df['passorfail']==0]
        neg_df = df[df['passorfail']==1]
        for i, col in enumerate(cols, 1):
            plt.subplot(n_rows, 3, i)
            plt.title(col)
            plt.hist(pos_df[col], label='OK')
            plt.hist(neg_df[col], label='NG')
            plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def plot_correlation_heatmap(self, df, save_path=None):
        """상관관계 히트맵 (가이드북 스타일)"""
        plt.figure(figsize=(15,15))
        sns.heatmap(data=df.corr(), annot=True, fmt='.2f', linewidths=0.5, cmap='Blues')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()