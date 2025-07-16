import json
from pathlib import Path

import pandas as pd
from multimodal_model import MultimodalModel
from tqdm import tqdm

from constants import SAMPLES_PER_CLUSTER


def generate_cluster_descriptions(
    cluster_mapping: Path, 
    video_description_file: str,
    output_dir: Path,
    samples_per_cluster: int = SAMPLES_PER_CLUSTER
) -> None:
    """Генерирует текстовые описания для кластеров

    Parameters
    ----------
    cluster_mapping : Path
        путь к файлу с маппингом кластеров
    video_description_file : str
        путь к файлу с текстовыми описаниями
    output_dir : Path
        директория для сохранения результатов
    samples_per_cluster : int, optional
        количество примеров для каждого кластера, по умолчанию SAMPLES_PER_CLUSTER
    """
    model = MultimodalModel()
    cluster_df = pd.read_csv(cluster_mapping)
    with open(video_description_file, 'r', encoding='utf-8') as file:
        descriptions = json.load(file)
    cluster_descriptions = {}
    cluster_prompts = {}
    for cluster_id in tqdm(cluster_df['cluster_id'].unique(), desc="Generating descriptions"):
        cluster_videos = cluster_df[cluster_df['cluster_id'] == cluster_id]['video_id']
        sample_videos = cluster_videos.sample(min(samples_per_cluster, len(cluster_videos)))
        
        context_descriptions = []
        for video_id in sample_videos:
            try:
                response = descriptions[video_id] 
                context_descriptions.append(response)
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
        
        # Создаем промпт для генерации описания кластера
        cluster_prompt = (
            " ".join([f"video № {i}- {desc}" for i, desc in enumerate(context_descriptions)]) +
              "Твоя задача написать тему не более 5 слов, которая связывает большинство видео например: Это видео на спортивную тематику" 
        )
        
        # Генерируем финальное описание кластера
        cluster_description = model.generate_response(
            frames=[], 
            audio_path=None, 
            text=cluster_prompt,
            max_new_tokens=100,
            temperature=0.13,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.17,
            top_k=5,
        )
        cluster_descriptions[str(cluster_id)] = cluster_description
        cluster_prompts[str(cluster_id)] = cluster_prompt
        
    
    with open(output_dir / 'cluster_descriptions.json', 'w') as f:
        json.dump(cluster_descriptions, f, indent=2, ensure_ascii=False)
        
    with open(output_dir / 'cluster_promts.json', 'w') as f:
        json.dump(cluster_prompts, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mapping', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--video_description_file', type=str, default='./output/video_description.json')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    generate_cluster_descriptions(
        cluster_mapping=Path(args.cluster_mapping),
        video_description_file=args.video_description_file,
        output_dir=output_dir
    )