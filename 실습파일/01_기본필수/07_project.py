"""실습 07~08 기본필수: 같은 파일을 완성하고 다시 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 CSV를 읽고, 전날과 같은 요청들을 12 / 4 / 4로 나눕니다.

# 현재 파이썬 파일이 있는 폴더 경로
HERE = Path(__file__).resolve().parent
# requests.csv 파일 읽기
raw = pd.read_csv(HERE / "requests.csv")
# 전체 데이터의 20%를 test로 분리하고, 나머지 80%를 develop으로 저장
# failed 비율이 유지되도록 stratify 사용
develop, test = train_test_split(raw, test_size=0.2, random_state=42, stratify=raw["failed"])
# develop의 25%를 valid로 분리
# 결과적으로 train:valid:test = 60%:20%:20% (12:4:4)
train, valid = train_test_split(develop, test_size=0.25, random_state=42, stratify=develop["failed"])
# test는 보관만 합니다. 학습·예측에 넣지 않습니다.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score

# 제공: 전날의 두 입력을 그대로 사용합니다. elapsed_ms와 failed는 정답이므로 입력에서 뺍니다.

# 입력 변수(features) 설정
# elapsed_ms와 failed는 정답(label)이므로 입력에서 제외
features = ["input_chars", "task_type"]
# 학습용 입력 데이터
X_train = train[features]
# 검증용 입력 데이터
X_valid = valid[features]
# 전처리 정의
# 숫자형(input_chars)은 StandardScaler로 표준화
# 범주형(task_type)은 OneHotEncoder로 분류
pre = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])
# 제공: 아래 두 반복은 같은 구조입니다. 첫 번째는 숫자, 두 번째는 클래스 정답을 사용합니다.

# 회귀 모델 평가 결과(MAE) 저장용 리스트
regression_rows = []
# 분류 모델 평가 결과(F1 Score) 저장용 리스트
classification_rows = []
# 검증 데이터의 실제값과 예측값을 함께 저장할 결과 데이터프레임
result = valid[["request_id", "elapsed_ms", "failed"]].copy()

# 회귀 모델 비교
# DummyRegressor와 LinearRegression의 성능 비교
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression())]:
    # 전처리 + 모델을 하나의 파이프라인으로 구성
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO B07a: 실행시간 elapsed_ms로 학습하고 X_valid를 예측하세요.
    # 힌트: model.fit(X_train, train["elapsed_ms"]) 다음 predict(X_valid)를 연결합니다.
    # pass

    # elapsed_ms를 목표값으로 학습
    model.fit(X_train, train["elapsed_ms"])
    # 검증 데이터 예측
    prediction = model.predict(X_valid)
    
    # MAE 계산 후 저장
    regression_rows.append({"model": name, "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction)})
    # 예측 결과 저장
    result[name + "_ms"] = prediction

# 분류 모델 비교
# DummyClassifier와 LogisticRegression의 성능 비교
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(max_iter=1000))]:
    # 전처리 + 모델 파이프라인 구성
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO B07b: 실패 여부 failed로 학습하고 X_valid를 예측하세요.
    # 힌트: 회귀와 같은 흐름이며 정답 열만 failed로 바뀝니다.
    # pass

    # failed를 목표값으로 학습
    model.fit(X_train, train["failed"])
    # 검증 데이터 예측
    prediction = model.predict(X_valid)
    
    # F1 Score 계산 후 저장
    classification_rows.append({"model": name, "F1": f1_score(valid["failed"], prediction, zero_division=0)})
    # 예측 결과 저장
    result[name + "_failed"] = prediction

# 제공: 두 종류의 지표를 섞지 않고 별도 표로 저장합니다. 저장 코드는 수정하지 않아도 됩니다.
# 회귀 모델 성능 비교표 생성
regression = pd.DataFrame(regression_rows)
# 분류 모델 성능 비교표 생성
classification = pd.DataFrame(classification_rows)
# 결과 CSV 저장
regression.to_csv(HERE / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
result.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
# 결과 출력
print("분할:", len(train), len(valid), len(test))
print("회귀 비교 / 낮은 MAE가 좋음")
print(regression.round(3).to_string(index=False))
print("분류 비교 / 높은 F1이 좋음")
print(classification.round(3).to_string(index=False))
print("저장: regression_comparison.csv, classification_comparison.csv, validation_predictions.csv")
print("test는 학습·예측하지 않았습니다.")
