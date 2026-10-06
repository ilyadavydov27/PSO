# PSO Watermarking: влияние состава целевой функции на эффективность

Исследование влияния набора атак в целевой функции на эффективность 
оптимизации параметров встраивания цифровых водяных знаков методом 
роя частиц (PSO).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## О проекте

Цифровые водяные знаки должны быть устойчивы к атакам: сжатию, шуму, 
фильтрации и другим. При подборе параметров встраивания метаэвристиками 
(PSO) возникает вопрос: как выбор набора атак в целевой функции влияет 
на итоговую робастность и незаметность?

В проекте реализован алгоритм встраивания ЦВЗ с PSO-оптимизацией и 
проведены эксперименты:

- Официальная абляция — воспроизведение экспериментов из статьи с 
  варьированием числа атак в целевой функции.
- Исследование специализации — оценка переноса робастности на атаки, 
  не входившие в целевую функцию.

## Возможности

- Arnold Cat Map для перемешивания водяного знака.
- PSO-оптимизация параметров встраивания.
- Набор атак на контейнер (сжатие, шум, фильтрация и др.).
- Метрики: PSNR, SSIM (незаметность), BER, NC (робастность).
- Скрипты для воспроизведения экспериментов с выгрузкой в Excel.

## Структура проекта

praktos_watermarking/
├── arnold.py
├── attacks.py
├── metrics.py
├── pso_watermarking.py
├── run_all_experiments.py
├── run_official_ablation.py
├── run_specialization_study.py
├── requirements.txt
├── article.pdf
├── pics/
├── results/
└── README.md

## Установка

```bash
git clone https://github.com/ilyadavydov27/PSO.git
cd PSO
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt

## Использование

python run_all_experiments.py
python run_official_ablation.py
python run_specialization_study.py

## Результаты

Итоговые таблицы с метриками PSNR, SSIM, BER, NC при разных составах
целевой функции находятся в results/.