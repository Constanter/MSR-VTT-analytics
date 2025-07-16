import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger


class VideoProcessor:
    def _extract_audio(self, video_path: Path, audio_path: Path) -> bool:
        """Извлеккает аудио из видео и сохраняет его в файл.
    
        Parameters
        ----------
        video_path : Path
            Путь к видео.
        audio_path : Path
            Путь к аудио

        Returns
        -------
        bool
            True, если успешно, False в случае ошибки.
        """
        try:
            cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-ac', '1', '-ar', '16000', str(audio_path)
            ]
    
            
            # Run the extraction
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio extraction failed: {video_path.stem} --- {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Error getting video duration: {str(e)}")
            return False

    def extract_frames(self, video_path: Path, num_frames: int = 5, target_height: int = 360) -> List[Path]:
        """Извлекает кадры из видео.

        Parameters
        ----------
        video_path : Path
            Путь к видео файлу.
        num_frames : int, optional
            Количество кадров для извлечения, по умолчанию 5
        target_height : int, optional
            Целевое значение разрешения изображения, по умолчанию 360

        Returns
        -------
        List[Path]
            Список путей к извлеченным кадрамм.
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                new_w = int(w * target_height / h)
                frame = cv2.resize(frame, (new_w, target_height))
                frame_path = f"/tmp/frame_{video_path.stem}_{idx}.jpg"
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frames.append(Path(frame_path))
        cap.release()
        return frames
    
    def extract_audio(self, video_path: Path) -> Path | None:
        """
        Извлекает аудио из видео-файла.
        
        Parameters
        ----------
        video_path : Path
            Путь к видео-файлу.

        Returns
        -------
        Path | None
           Путь к аудио-файлу.
        """
        audio_path = Path("./tmp.wav")
        audio_success = self._extract_audio(video_path, audio_path)
        
        return Path(audio_path) if audio_success else None
    
    def process_video(self, video_path: Path, num_frames: int = 5, target_height: int = 360) -> Tuple[Path | None, List[Path]]:
        """
        Процессор видео-файла.

        Parameters
        ----------
        video_path : Path
            Путь к видео-файлу.
        num_frames : int, optional
            Количество кадров для извлечения, по умолчанию 5
        target_height : int, optional
            Целевое значение размера изображения, по умолчанию 360

        Returns
        -------
        Tuple[Path | None, List[Path]]
           Путь к аудио-файлу и список путей к изображения
        """
        
        audio_path = self.extract_audio(video_path)
        frames = self.extract_frames(video_path, num_frames, target_height)
        return audio_path, frames