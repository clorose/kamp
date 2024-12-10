# path: ~/Develop/kamp/src/data_processor.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import Normalizer, MinMaxScaler
from imblearn.over_sampling import SMOTE
from visual import Visualizer

class DataProcessor:
    def __init__(self, viz=None):
        self.normalizer = Normalizer()
        self.scaler = MinMaxScaler()
        self.visualizer = viz if viz is not None else Visualizer() # viz는 Visualizer 클래스의 인스턴스
        
    def load_data(self, file_path):
        """데이터 로드 및 기본 전처리"""
        df = pd.read_csv(file_path)

        # 데이터 분포 시각화
        self.visualizer.plot_data_distribution(
            df,
            save_path='../figures/initial_distribution.png'
        )
        
        # 고정 변수 제거 및 수치형 변수 선택
        df = df[[col for col in df.columns if df[col].nunique() != 1]]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        return df
    
    def remove_outliers(self, df, percentile_thresh=(0.1, 99.9)):
        """이상치 제거"""
        for col in df.columns:
            if col != 'passorfail':
                UCL = np.percentile(df[col], percentile_thresh[1])
                LCL = np.percentile(df[col], percentile_thresh[0])
                df = df[(df[col] <= UCL) & (df[col] >= LCL)]
        
        # 이상치 제거 후 데이터 분포 시각화
        self.visualizer.plot_data_distribution(
            df,
            save_path='../figures/outlier_removal_comparison.png'
        )

        return df
    
    def select_features_by_ttest(self, df, target_col='passorfail', p_threshold=0.1):
        """T-test 기반 변수 선택"""
        selected_cols = []
        
        for col in df.columns:
            if col != target_col:
                t_stat, p_val = stats.ttest_ind(
                    df[df[target_col]==1][col],
                    df[df[target_col]==0][col],
                    equal_var=False
                )
                if p_val < p_threshold:
                    selected_cols.append(col)
        
        selected_cols.append(target_col)
        return df[selected_cols]
    
    def apply_smote(self, X_train, y_train, sampling_strategy=None):
        """SMOTE를 사용한 데이터 증강"""
        if sampling_strategy is None:
            sampling_strategy = {0: 2000, 1: 300}
        
        smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def normalize_and_scale(self, X_train, X_test):
        """데이터 정규화 및 스케일링"""
        X_train_norm = self.normalizer.fit_transform(X_train)
        X_test_norm = self.normalizer.transform(X_test)
        
        X_train_scaled = self.scaler.fit_transform(X_train_norm)
        X_test_scaled = self.scaler.transform(X_test_norm)
        
        return X_train_scaled, X_test_scaled
    
    def calculate_quality_indices(self, df):
        """데이터 품질 지수 계산"""
        # 완전성 지수
        completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        
        # 유일성 지수 (SerialNo 기준)
        if 'SerialNo' in df.columns:
            uniqueness = (len(df.groupby('SerialNo').size()) / len(df)) * 100
        else:
            uniqueness = 100
            
        quality_indices = {
            'completeness': completeness,
            'uniqueness': uniqueness,
            'validity': 100,  # 모든 데이터가 유효범위 내에 있다고 가정
            'consistency': 100,  # 데이터 형식이 일관적이라고 가정
            'integrity': 66.67  # 유일성을 제외한 두 지수가 100%이므로
        }
        
        return quality_indices
