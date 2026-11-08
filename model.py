import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('data/stroke.csv')
x = df.drop('stroke', axis=1)
y = df['stroke']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

num_features = ['age', 'avg_glucose_level', 'bmi']
cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced',
        max_iter=2000,
        random_state=42
    ))
])

print("Обучение модели...")
model.fit(x_train, y_train)
print("Готово!")

y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]
CUSTOM_THRESHOLD = 0.75
y_pred_custom = (y_proba >= CUSTOM_THRESHOLD).astype(int)

print(f"--- Отчет с порогом {CUSTOM_THRESHOLD} ---")
print(classification_report(y_test, y_pred_custom))

joblib.dump(model, 'models/stroke_model.pkl')
print("Модель успешно сохранена!")