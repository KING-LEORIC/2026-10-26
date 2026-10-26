"""실습 07~08 응용: 한 파일에서 회귀·분류 각각 세 후보를 비교합니다."""
from pathlib import Path
import pandas as pd

# 현재 파이썬 파일이 있는 폴더 경로 저장
HERE = Path(__file__).resolve().parent
# 원본 데이터 읽기
# request_id를 인덱스로 설정하여 ID를 기준으로 쉽게 조회할 수 있게 함
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")
# train에 속한 request_id만 원래 순서대로 추출
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
# valid에 속한 request_id만 원래 순서대로 추출
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
# train ID에 해당하는 데이터만 선택
train = raw.loc[train_ids]
# valid ID에 해당하는 데이터만 선택
valid = raw.loc[valid_ids]
# 모델에 사용할 입력 변수(특성) 목록
features = ["input_chars", "allowed_tools", "task_type", "priority"]
# 학습용 입력 데이터
X_train = train[features]
# 검증용 입력 데이터
X_valid = valid[features]
# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.

# 숫자형 변수 처리 파이프라인
# 결측치는 중앙값으로 채우고(StandardScaler 전에)
# 이후 평균 0, 표준편차 1 형태로 표준화
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
# 범주형 변수 처리 파이프라인
# 결측치는 최빈값으로 채우고
# 범주를 원-핫 인코딩으로 변환
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
# 숫자형/범주형 변수에 서로 다른 전처리 적용
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# 제공 K: PPT 실습06의 동일 원본·분할·standard 탐색 결과(분류3, 회귀15)를 재사용합니다.

# 회귀 모델 평가 결과 저장
regression_rows = []
# 분류 모델 평가 결과 저장
classification_rows = []
# 각 모델의 예측값 저장
predictions = []
# 회귀 모델 3종 비교
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression()),
                        ("KNN15", KNeighborsRegressor(n_neighbors=15))]:
    # 전처리 + 모델을 하나의 파이프라인으로 묶음
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07r: 회귀 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: 정답 열은 elapsed_ms입니다.
    # pass

    # elapsed_ms를 목표값으로 학습
    model.fit(X_train, train["elapsed_ms"])
    # valid 데이터 예측
    prediction = model.predict(X_valid)

    # MAE, RMSE 계산 후 저장
    regression_rows.append({"model": name, "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})
    # 실제값과 예측값 저장
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "regression", "model": name,
        "actual": valid["elapsed_ms"].to_numpy(), "predicted": prediction}))

# 분류 모델 3종 비교
for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                        ("KNN3", KNeighborsClassifier(n_neighbors=3))]:
    # 전처리 + 모델을 하나의 파이프라인으로 묶음
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07c: 분류 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: 정답 열은 failed입니다.
    # pass
    
    # failed 여부를 목표값으로 학습
    model.fit(X_train, train["failed"])
    # valid 데이터 예측
    prediction = model.predict(X_valid)

    # 혼동행렬 생성
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()
    # 분류 성능 지표 저장
    classification_rows.append({"model": name, "scale": "standard",
        "F1": f1_score(valid["failed"], prediction, zero_division=0),
        "Precision": precision_score(valid["failed"], prediction, zero_division=0),
        "Recall": recall_score(valid["failed"], prediction, zero_division=0), "FP": fp, "FN": fn})
    # 실제값과 예측값 저장
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "classification", "model": name,
        "actual": valid["failed"].to_numpy(), "predicted": prediction}))

# 회귀 결과 데이터프레임 생성
regression = pd.DataFrame(regression_rows)
# 분류 결과 데이터프레임 생성
classification = pd.DataFrame(classification_rows)
# 결과 저장 폴더 생성
out = HERE / "outputs"
out.mkdir(exist_ok=True)

# 회귀 비교 결과 저장
regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
# 분류 비교 결과 저장
classification.to_csv(out / "classification_comparison.csv", index=False, encoding="utf-8-sig")
# 모든 모델의 예측 결과를 하나로 합쳐 저장
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")

# 결과 출력
print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
