# path: ~/Develop/kamp/src/main.py
from data_processor import DataProcessor
from model_handler import ModelHandler
from visual import Visualizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

def main():
    # 경로 설정
    DATA_PATH = "../data/정밀가공_품질보증_데이터셋.csv"
    
    # 실험 결과를 위한 타임스탬프 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = f"../runs/{timestamp}"
    MODEL_DIR = f"../models/{timestamp}"
    FIGURE_DIR = f"../figures/{timestamp}"

    for dir_path in [RUN_DIR, MODEL_DIR, FIGURE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # 데이터 처리 및 시각화 객체 초기화
    processor = DataProcessor()
    visualizer = Visualizer()

    # 1. 데이터 로드 및 초기 시각화
    print("Loading data and visualizing initial distributions...")
    df = processor.load_data(DATA_PATH)
    
    # 가이드북 Figure 2: 초기 변수별 분포
    visualizer.plot_initial_distributions(df, save_path=os.path.join(FIGURE_DIR, "initial_distributions.png"))
    
    # 가이드북 Figure 3: 양품/불량별 분포
    visualizer.plot_class_distributions(df, save_path=os.path.join(FIGURE_DIR, "class_distributions.png"))

    # 2. 데이터 전처리
    print("Preprocessing data...")
    # 고정 변수 제거 및 수치형 변수 선택
    df = df[[col for col in df.columns if df[col].nunique() != 1]]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numeric_cols]

    # 이상치 제거
    df = processor.remove_outliers(df)
    
    # T-test 기반 변수 선택
    df_ttest = processor.select_features_by_ttest(df)
    
    # 가이드북 Figure 4: 선택된 변수들의 상관관계 히트맵
    visualizer.plot_correlation_heatmap(df_ttest, save_path=os.path.join(FIGURE_DIR, "correlation_heatmap.png"))

    # 3. 학습/테스트 데이터 분할
    print("Splitting data and applying SMOTE...")
    X = df_ttest.drop('passorfail', axis=1)
    y = df_ttest['passorfail']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # SMOTE 적용
    X_train_resampled, y_train_resampled = processor.apply_smote(
        X_train, y_train, sampling_strategy={0: 2000, 1: 300}
    )

    # SMOTE 전후 비교 시각화
    visualizer.plot_preprocessing_comparison(
        pd.DataFrame({'passorfail': y_train}),
        pd.DataFrame({'passorfail': y_train_resampled}),
        save_path=os.path.join(FIGURE_DIR, "smote_comparison.png")
    )

    # 4. 데이터 정규화
    print("Normalizing and scaling data...")
    X_train_scaled, X_test_scaled = processor.normalize_and_scale(
        X_train_resampled, X_test
    )

    # numpy array로 변환
    X_train_scaled = np.array(X_train_scaled)
    X_test_scaled = np.array(X_test_scaled)
    y_train_resampled = np.array(y_train_resampled)
    y_test = np.array(y_test)

    # 5. 모델 학습 및 평가
    print("Training and evaluating model...")
    model_handler = ModelHandler(model_dir=MODEL_DIR, viz=visualizer, figure_dir=FIGURE_DIR)
    fold_metrics = model_handler.train_with_kfold(
        X_train_scaled, y_train_resampled, X_test_scaled, y_test
    )

    # 결과 저장
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj

    results = {
        "timestamp": timestamp,
        "fold_metrics": [
            {k: convert_to_json_serializable(v) for k, v in metrics.items()}
            for metrics in fold_metrics
        ],
        "average_metrics": {
            "accuracy": convert_to_json_serializable(
                np.mean([m["accuracy"] for m in fold_metrics])
            ),
            "f1_score": convert_to_json_serializable(
                np.mean([m["f1"] for m in fold_metrics])
            ),
        },
        "data_shape": {
            "original": convert_to_json_serializable(df.shape),
            "after_smote": convert_to_json_serializable(X_train_resampled.shape),
        },
        "class_distribution": {
            "original": convert_to_json_serializable(y.value_counts()),
            "after_smote": convert_to_json_serializable(
                pd.Series(y_train_resampled).value_counts()
            ),
        },
    }

    # 결과 저장
    with open(os.path.join(RUN_DIR, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"Experiment results saved to: {RUN_DIR}")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Average F1 Score across folds: {results['average_metrics']['f1_score']:.4f}")

if __name__ == "__main__":
    main()