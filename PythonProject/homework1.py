import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import os

# Фиксируем seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

#шаг 1 Загрузка и исследование данных
#читаем train (с метками) и test_features (без меток).
#Смотрим размер, структуру, пропуски и баланс классов без этого нельзя понять, как чистить данные и какую метрику выбрать.
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_features.csv')

print("Размер train:", train_df.shape)
print("Размер test:", test_df.shape)
print("\nПропуски в train:")
print(train_df.isnull().sum().sort_values(ascending=False).head(10))

#Целевая переменная Доклинический риск (0 или 1).
#Столбцы prediction и Статус_глюкозы убираем из признаков: это тоже метки, модель не должна их видеть иначе подсмотрит ответ.
target_col = 'Доклинический_риск'
cols_to_drop = ['patient_id', 'prediction', 'Статус_глюкозы', 'Доклинический_риск']

X = train_df.drop(columns=cols_to_drop)
y = train_df[target_col].values

test_ids = test_df['patient_id']
# Приводим колонки теста к тому же набору и порядку, что и в train, иначе scaler и модель упадут с ошибкой несовпадения признаков.
X_test = test_df.drop(columns=['patient_id'], errors='ignore')[X.columns]

print("\nбаланс классов в train")
print(pd.Series(y).value_counts(normalize=True))


#шаг 2 Чистка данных
#убираем выбросы через IQR: они смещают веса нейросети и мешают обобщению. Применяем ТОЛЬКО к train тест трогать нельзя
def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    return df

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
#склеиваем X и y, чтобы при удалении строк связь признака и ответа не порвалась
train_combined = pd.concat([X, pd.Series(y, name=target_col)], axis=1)
train_combined = remove_outliers_iqr(train_combined, numeric_cols)

X = train_combined.drop(columns=[target_col])
y = train_combined[target_col].values
print(f"\nразмер после удаления выбросов: {X.shape}")

#заполняем пропуски средним. Среднее считаем только на train иначе будет Data Leakage тест просочится в обучение
train_means = X.mean()
X = X.fillna(train_means)
X_test = X_test.fillna(train_means)

#3.нормализация и разделение
#Нейросеть лучше учится, когда все признаки одного масштаба. StandardScaler приводит их к среднему 0 и std 1
#fit только на train, transform, на train/val/test отдельно
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y)
X_val, X_test_inner, y_val, y_test_inner = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_inner = scaler.transform(X_test_inner)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test_inner.shape}")


#шаг4.Обучение нейросети (1 слой, 32 нейрона, ReLU, 150 эпох)
#Конвертируем numpy в тензоры PyTorch. y приводим к форме (N, 1), чтобы совпадала с выходом сети
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).view(-1, 1)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.FloatTensor(y_val).view(-1, 1)

#DataLoader режет данные на батчи по 32 и перемешивает их каждую эпоху. Перемешивание важно потому что без него сеть запомнит порядок и переобучится.
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=32, shuffle=True)

#архитектура 1 скрытый слой с 32 нейронами и ReLU, затем выходной слой с сигмоидой (даёт вероятность от 0 до 1).
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 32),   #1 слой, 32 нейрона
    nn.ReLU(),                         #ReLU
    nn.Linear(32, 1),
    nn.Sigmoid()                       #бинарная классификация
)

#BCELoss это стандартная функция потерь для бинарной классификации. Adam это оптимизатор
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses = [], []
epochs = 150                            #150 эпох

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()                    #обнуляем старые градиенты
        loss = criterion(model(X_batch), y_batch)  #прямой проход и ошибка
        loss.backward()                          #обратный проход
        optimizer.step()                         #шаг оптимизатора
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

#валидация — в режиме eval и без расчёта градиентов.
    model.eval()
    with torch.no_grad():
        val_losses.append(criterion(model(X_val_t), y_val_t).item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

#шаг5.метрики на валидации
#accuracy одна не годится при дисбалансе классов нужно смотреть ещё F1 и AUC.
model.eval()
with torch.no_grad():
    y_val_proba = model(X_val_t).numpy().flatten()
    y_val_pred = (y_val_proba > 0.5).astype(int)

print("\nМетрики на валидации")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"F1-score: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_val, y_val_proba):.4f}")
print("\nClassification report:")
print(classification_report(y_val, y_val_pred))

#шаг6.Визуализация обучения
#смотрим, как падает loss. Если val_loss растёт, а train_loss падает это переобучение, и стоит уменьшить число эпох.
plt.figure(figsize=(10, 5))
sns.lineplot(x=range(1, epochs+1), y=train_losses, label='Train Loss')
sns.lineplot(x=range(1, epochs+1), y=val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/loss_plot.png', dpi=150)
plt.show()

#шаг7.предсказания и сохранение submission
#порог 0.5 превращает вероятность в класс: > 0.5 в 1, иначе 0.
model.eval()
with torch.no_grad():
    predictions = (model(torch.FloatTensor(X_test_scaled)).numpy().flatten() > 0.5).astype(int)

#сохраняем результат в outputs/task1_predictions.csv
os.makedirs('outputs', exist_ok=True)
submission = pd.DataFrame({
    'patient_id': test_ids,
    'prediction': predictions
})
submission.to_csv('outputs/task1_predictions.csv', index=False)