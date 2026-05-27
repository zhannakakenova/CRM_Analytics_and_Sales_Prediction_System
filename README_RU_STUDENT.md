# CRM проект по анализу данных

Это простой школьный проект по Data Science.

Мы работаем с данными продаж AdventureWorks и хотим понять:

- что продавалось лучше всего;
- какие клиенты важные;
- какие регионы приносят больше денег;
- купит ли клиент товар или нет;
- сколько примерно будет продаж.

## Главная идея проекта

Представь, что у нас есть магазин велосипедов и товаров для велосипедов.

У магазина много данных:

- кто покупал;
- какой товар покупали;
- в какой стране или регионе;
- сколько товаров купили;
- сколько денег магазин получил.

Наша задача: посмотреть на эти данные, найти интересные факты и построить две модели машинного обучения.

## Что требует задание

По файлу `tasks.md` нужно сделать 5 вещей:

1. Очистить данные.
2. Сделать EDA, то есть исследовать данные.
3. Найти интересные бизнес-выводы.
4. Построить две модели машинного обучения.
5. Показать результат красиво, например через Gradio dashboard.

## Какие модели используются

В проекте есть две модели.

### 1. Classification

Файл:

```bash
src/predict_classification.py
```

Вопрос модели:

```text
Купит клиент этот товар или не купит?
```

Ответ модели:

```text
Buy
```

или

```text
Not Buy
```

Используется модель:

```text
RandomForestClassifier
```

### 2. Regression

Файл:

```bash
src/predict_regression.py
```

Вопрос модели:

```text
Сколько денег продаж мы ожидаем?
```

Ответ модели:

```text
Например: 1024.52
```

Используется модель:

```text
RandomForestRegressor
```

## Структура проекта

### `tasks.md`

Это описание задания.

Там написано, что нужно сделать:

- очистка данных;
- анализ данных;
- prediction;
- Random Forest модели;
- презентация на 5 минут.

### `data/`

Это папка с данными.

Внутри есть таблицы:

- клиенты;
- товары;
- продажи;
- регионы;
- даты.

### `src/common.py`

Это общий помощник для проекта.

Он делает важные простые вещи:

- загружает CSV файлы;
- соединяет таблицы вместе;
- чистит данные;
- сохраняет обработанные данные.

### `src/data_exploration.py`

Этот файл делает EDA.

EDA означает:

```text
Exploratory Data Analysis
```

По-русски:

```text
исследовательский анализ данных
```

Этот файл:

- очищает данные;
- удаляет дубликаты;
- удаляет пропуски;
- убирает выбросы;
- создает графики.

После запуска появляются картинки в папке:

```bash
outputs/eda/
```

Например:

- продажи по месяцам;
- продажи по регионам;
- топ товаров;
- топ клиентов.

### `src/train_models.py`

Этот файл обучает модели.

Он обучает сразу две модели:

- `RandomForestClassifier`;
- `RandomForestRegressor`.

После запуска модели сохраняются в папку:

```bash
models/
```

### `src/predict_classification.py`

Это отдельный prediction pipeline для classification.

Он отвечает на вопрос:

```text
Купит клиент товар или нет?
```

### `src/predict_regression.py`

Это отдельный prediction pipeline для regression.

Он отвечает на вопрос:

```text
Сколько будет продаж?
```

### `src/dashboard_gradio.py`

Это главный dashboard.

Dashboard открывается в браузере.

В нем есть вкладки:

- `EDA Summary`;
- `Interactive Data Analysis`;
- `Classification: Buy / Not Buy`;
- `Regression: Sales Amount`.

## Как запустить проект

Сначала открой терминал в папке проекта.

## Шаг 1. Создать virtual environment

Virtual environment помогает хранить библиотеки только для этого проекта.

Команда:

```bash
python3 -m venv .venv
```

## Шаг 2. Включить virtual environment

Команда:

```bash
source .venv/bin/activate
```

Если все хорошо, в терминале появится:

```text
(.venv)
```

## Шаг 3. Установить библиотеки

Команда:

```bash
pip install -r requirements.txt
```

## Шаг 4. Сделать анализ данных

Команда:

```bash
python3 src/data_exploration.py
```

После этого появятся:

- очищенный файл данных;
- графики;
- summary JSON.

## Шаг 5. Обучить модели

Команда:

```bash
python3 src/train_models.py
```

После этого появятся файлы моделей:

```bash
models/random_forest_classifier.joblib
models/random_forest_regressor.joblib
```

## Шаг 6. Проверить classification prediction

Команда:

```bash
python3 src/predict_classification.py \
  --customer_key 11000 \
  --product_key 214 \
  --order_quantity 1 \
  --list_price 1200 \
  --country_region "United States" \
  --state_province "California" \
  --category "Bikes" \
  --subcategory "Mountain Bikes" \
  --color "Black" \
  --channel "Reseller" \
  --region "Northwest"
```

Ты увидишь примерно такой ответ:

```text
Result: Buy
Probability of Buy: 0.810
```

Это значит:

```text
Модель думает, что клиент купит товар.
```

## Шаг 7. Проверить regression prediction

Команда:

```bash
python3 src/predict_regression.py \
  --customer_key 11000 \
  --product_key 214 \
  --order_quantity 1 \
  --unit_price 1000 \
  --unit_price_discount_pct 0.0 \
  --list_price 1200 \
  --country_region "United States" \
  --state_province "California" \
  --category "Bikes" \
  --subcategory "Mountain Bikes" \
  --color "Black" \
  --channel "Reseller" \
  --region "Northwest" \
  --month "January" \
  --fiscal_quarter "Q1"
```

Ты увидишь примерно такой ответ:

```text
Predicted sales_amount: 1024.52
```

Это значит:

```text
Модель ожидает продажи примерно на 1024.52.
```

## Шаг 8. Открыть Gradio dashboard

Команда:

```bash
python3 src/dashboard_gradio.py
```

Потом открой в браузере:

```text
http://127.0.0.1:7960
```

## Что есть в Gradio dashboard

### EDA Summary

Здесь можно увидеть готовые графики:

- продажи по месяцам;
- продажи по регионам;
- топ товаров;
- топ клиентов.

### Interactive Data Analysis

Здесь можно самому исследовать данные.

Например, можно выбрать:

- продажи по странам;
- продажи по категориям;
- продажи по регионам;
- продажи по клиентам;
- продажи по месяцам.

Можно использовать фильтры:

- страна;
- категория;
- канал продаж.

### Classification: Buy / Not Buy

Здесь можно ввести данные клиента и товара.

Dashboard покажет:

- `Buy`;
- или `Not Buy`;
- вероятность покупки.

### Regression: Sales Amount

Здесь можно ввести данные товара, региона и цены.

Dashboard покажет:

```text
сколько продаж ожидает модель
```

## Простая последовательность запуска

Если коротко, запускай так:

```bash
source .venv/bin/activate
python3 src/data_exploration.py
python3 src/train_models.py
python3 src/dashboard_gradio.py
```

## Что сказать на презентации

Можно рассказать так:

1. Я взял данные AdventureWorks.
2. Я очистил данные: удалил пропуски, дубликаты и выбросы.
3. Я сделал EDA и посмотрел продажи по времени, регионам, клиентам и товарам.
4. Я построил classification модель, чтобы предсказывать `Buy / Not Buy`.
5. Я построил regression модель, чтобы предсказывать сумму продаж.
6. Я сделал Gradio dashboard, чтобы удобно показать результаты.

## Очень коротко

Этот проект отвечает на два главных вопроса:

```text
Купит ли клиент товар?
```

и

```text
Сколько денег может принести продажа?
```

Это полезно для CRM, потому что компания может лучше понимать клиентов и планировать продажи.
