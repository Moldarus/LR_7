import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Загрузка данных
data = pd.read_csv('power_usage_2016_to_2020.csv')
data['StartDate'] = pd.to_datetime(data['StartDate'])
data.set_index('StartDate', inplace=True)

# Генерация признаков
data['hour'] = data.index.hour
data['month'] = data.index.month
data['year'] = data.index.year
data['day_of_year'] = data.index.dayofyear
data['week_of_year'] = data.index.isocalendar().week
data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
data['is_holiday'] = data['notes'].apply(lambda x: 1 if x == 'holiday' else 0)

# Вычисление скользящего среднего
data['rolling_mean'] = data['Value (kWh)'].rolling(window=10).mean()

# Визуализация признаков
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

sns.lineplot(data=data, x=data.index, y='Value (kWh)', ax=axes[0, 0])
axes[0, 0].set_title('Value (kWh) over Time')

sns.histplot(data['Value (kWh)'], bins=50, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Value (kWh)')

sns.scatterplot(data=data, x='hour', y='Value (kWh)', ax=axes[1, 0])
axes[1, 0].set_title('Value (kWh) by Hour')

sns.scatterplot(data=data, x='month', y='Value (kWh)', ax=axes[1, 1])
axes[1, 1].set_title('Value (kWh) by Month')

sns.scatterplot(data=data, x='day_of_year', y='Value (kWh)', ax=axes[2, 0])
axes[2, 0].set_title('Value (kWh) by Day of Year')

sns.scatterplot(data=data, x='week_of_year', y='Value (kWh)', ax=axes[2, 1])
axes[2, 1].set_title('Value (kWh) by Week of Year')

plt.tight_layout()
plt.show()

# Визуализация скользящего среднего
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=data, x=data.index, y='rolling_mean', ax=ax)
ax.set_title('Rolling Mean Value (kWh) with Window Size 10')
plt.show()

# Исследование данных с точки зрения асимметрии и эксцесса
skewness = skew(data['Value (kWh)'])
kurt = kurtosis(data['Value (kWh)'])

print(f'Асимметрия: {skewness}')
print(f'Эксцесс: {kurt}')

# Визуализация асимметрии и эксцесса
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.histplot(data['Value (kWh)'], bins=50, kde=True, ax=axes[0])
axes[0].set_title(f'Асимметрия: {skewness}')

sns.histplot(data['Value (kWh)'], bins=50, kde=True, ax=axes[1])
axes[1].set_title(f'Эксцесс: {kurt}')

plt.tight_layout()
plt.show()

# График в графике для сравнения метрик
fig, ax1 = plt.subplots(figsize=(10, 5))

# Первый график
color = 'tab:red'
ax1.set_xlabel('Время')
ax1.set_ylabel('Значение (кВт*ч)', color=color)
ax1.plot(data.index, data['Value (kWh)'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Второй график
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Скользящее среднее', color=color)
ax2.plot(data.index, data['rolling_mean'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Сравнение значений (кВт*ч) и скользящего среднего')
plt.show()

# График в графике для сравнения метрик (гистограмма и плотность)
fig, ax1 = plt.subplots(figsize=(10, 5))

# Первый график
color = 'tab:red'
ax1.set_xlabel('Значение (кВт*ч)')
ax1.set_ylabel('Частота', color=color)
sns.histplot(data['Value (kWh)'], bins=50, kde=False, ax=ax1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Второй график
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Плотность', color=color)
sns.kdeplot(data['Value (kWh)'], ax=ax2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Сравнение частоты и плотности значений (кВт*ч)')
plt.show()

# График в графике для сравнения метрик (значения и скользящее среднее)
fig, ax1 = plt.subplots(figsize=(10, 5))

# Первый график
color = 'tab:red'
ax1.set_xlabel('Время')
ax1.set_ylabel('Значение (кВт*ч)', color=color)
ax1.plot(data.index, data['Value (kWh)'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Второй график
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Скользящее среднее', color=color)
ax2.plot(data.index, data['rolling_mean'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Сравнение значений (кВт*ч) и скользящего среднего')
plt.show()

# Аналитическая информация и описание результатов
if skewness > 0:
    print("Асимметрия положительная, что указывает на правостороннюю асимметрию. Это означает, что в данных есть длинный правый хвост, и большинство значений находятся слева от среднего значения.")
else:
    print("Асимметрия отрицательная, что указывает на левостороннюю асимметрию. Это означает, что в данных есть длинный левый хвост, и большинство значений находятся справа от среднего значения.")

if kurt > 3:
    print("Эксцесс положительный, что указывает на лептокуртическое распределение. Это означает, что распределение имеет более острый пик и более тяжелые хвосты по сравнению с нормальным распределением.")
else:
    print("Эксцесс отрицательный, что указывает на платикуртическое распределение. Это означает, что распределение имеет более плоский пик и более легкие хвосты по сравнению с нормальным распределением.")
