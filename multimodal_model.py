from pathlib import Path
from typing import List, Tuple

import soundfile as sf
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


class MultimodalModel:
    """
    Мультимодальная модель для обработки видео
    """
    def __init__(self, model_path: str = 'Lexius/Phi-4-multimodal-instruct'):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._load_model()
        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'
    
    def _load_model(self) -> None:
        logger.info(f"Loading model from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype='auto'
        ).to(self.device)
        
        try:
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_path,
                'generation_config.json'
            )
        except:
            self.generation_config = None
            logger.warning("No generation config found, using defaults")
    
    def prepare_inputs(
        self,
        frames: List[Path],
        audio_path: Path,
        text: str,
        max_audio_length: int = 461209
    ) -> Tuple[torch.Tensor, dict]:
        images = [Image.open(frame) for frame in frames]
        image_promt = "".join([f'<|image_{i+1}|>' for i, _ in enumerate(frames)])
        if audio_path:
            audio_array, rate = sf.read(audio_path)
            audio_array = audio_array[:max_audio_length]
            audio = (audio_array, rate)
        
       
        
            prompt = (f"{self.user_prompt}{image_promt}<|audio_1|>"
                    f"Use and images and audio to summarize what video is about"
                    f"{self.prompt_suffix}{self.assistant_prompt}")
            
            inputs = self.processor(
                text=prompt,
                images=images,
                audios=[audio],
                return_tensors='pt'
            ).to(self.device)
        else:
            prompt = (f"{self.user_prompt}{image_promt}"
                    f"Use and images to summarize what video is about"
                    f"{self.prompt_suffix}{self.assistant_prompt}")
            
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors='pt'
            ).to(self.device)
            
        
        return inputs
    
    def get_content_vector(
        self,
        frames: List[Path],
        audio_path: Path,
        text: str
    ) -> torch.Tensor:
        """ Получает контентный вектор изображений и аудио. 

        Parameters
        ----------
        frames : List[Path]
            Пути к изображениям из видео.
        audio_path : Path
            Путь к аудио.
        text : str
            Текстовое описание.

        Returns
        -------
        torch.Tensor
           Контентный вектор. 
        """
        inputs = self.prepare_inputs(frames, audio_path, text)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            last_hidden_state = outputs.hidden_states[-1]
            input_length = inputs.attention_mask.sum(dim=1)
            content_vector = last_hidden_state[0, input_length[0] - 1].cpu()
            
        return content_vector
    
    def generate_response(
        self,
        frames: List[Path],
        audio_path: Path,
        text: str,
        **generation_kwargs
    ) -> str:
        """Функция для генерации текстового ответа.

        Parameters
        ----------
        frames : List[Path]
            Путь к кадрам из видео.
        audio_path : Path
            Путь к аудио из видео.
        text : str
            Текстовое описание.

        Returns
        -------
        str
            Сгенерированный текстовый ответ
        """
        if not frames and not audio_path:
            inputs = self.processor(
                text=text,
                return_tensors='pt'
            ).to(self.device)
        else:
            inputs = self.prepare_inputs(frames, audio_path, text)
        
        gen_config = self.generation_config.to_dict() if self.generation_config else {}
        gen_config.update(generation_kwargs)
        
        generated_ids = self.model.generate(
            **inputs,
            generation_config=GenerationConfig(**gen_config),
            num_logits_to_keep=0
        )
        
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response.replace(text, "")
        response = response.replace("Use and images and audio to summarize what video is about", "")
        return response
    