import json
from pathlib import Path

from tqdm import tqdm

from constants import NUM_FRAMES, TARGET_HEIGHT
from multimodal_model import MultimodalModel
from video_processor import VideoProcessor

def extract_embeddings(video_dir: Path, output_dir: Path, subset_size: int = None) -> None:
    """Извлекает мультимодальные эмбеддинги для видео и делает текстовый описания для каждого видео.
    
    Parameters
    ----------
    video_dir : Path
        директория с видео
    output_dir : Path
        директория для сохранения результатов
    subset_size : int, optional
        количество видео для извлечения если хотим взять только часть видео (если None берем весь датасет), по умолчанию None
    """
    model = MultimodalModel()
    video_processor = VideoProcessor()
    
    video_paths = list(video_dir.glob('*.mp4'))
    if subset_size:
        video_paths = video_paths[:subset_size]
    
    embeddings = {}
    video_description = {}
    for video_path in tqdm(video_paths, desc="Extracting embeddings"):
        try:
            audio_path, frames = video_processor.process_video(
                video_path, 
                num_frames=NUM_FRAMES,
                target_height=TARGET_HEIGHT
            )
            embedding = model.get_content_vector(frames, audio_path, "")
            response = model.generate_response(frames, audio_path, "", max_new_tokens=500,)
            video_description[video_path.stem] = response
            embeddings[video_path.stem] = embedding.tolist()
            with open(output_dir / 'video_description.json', 'w') as f:
                json.dump(video_description, f)
                
            with open(output_dir / 'embeddings.json', 'w') as f:
                json.dump(embeddings, f)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--subset_size', type=int, default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    extract_embeddings(
        video_dir=Path(args.video_dir).absolute(),
        output_dir=output_dir,
        subset_size=args.subset_size
    )
