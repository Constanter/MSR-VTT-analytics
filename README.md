# MSR-VTT-analytics

Проект сделан для анализа и кластерризации датасета с видео MSR-VTT.

Процесс происходит в 3 стадии:
1. С помощью VLM Lexius/Phi-4-multimodal-instruct(берем 10 кадров из видео и аудио) извлекаем эмбеддинг и получаем описание видео.
2. Кластеризуем получившиеся эмбеддинги.
3. Для каждого кластера берем по 100 видео из кластера и вытаскиваем описание этих видео и делаем суммаризацию, обьединяя их общей темой. 

Инструкции по использованию:
1. Структура проекта:

project/
├── Dockerfile
├── requirements.txt
├── run_pipeline.py
├── extract_embeddings.py
├── cluster_embeddings.py
├── generate_descriptions.py
├── multimodal_model.py
└── video_processor.py
├── output/


2. Сборка Docker-образа:

```bash
docker build -t video-clustering .
```


3. Запуск пайплайна:

```bash
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/output:/app/output video-clustering \
    python run_pipeline.py \
    --video_dir /data/videos \
    --subset_size 1000 \
    --n_clusters 15 \
    --output_dir /app/output

```

4. Выходные файлы:

- video_cluster_mapping.csv - Соответствие видео кластерам

- cluster_descriptions.json - Текстовые описания кластеров

- cluster_promts.json - Набор видео описаний для генерации описания кластера

- tsne_visualization.png - Визуализация распределения кластеров

- cluster_distribution.png - Распределение видео по кластерам

- clustering_metrics.txt - Метрики качества кластеризации

- video_description.json - Текстовые описания видео