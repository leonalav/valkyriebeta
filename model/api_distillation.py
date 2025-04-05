import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import json
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITeacherModel:
    """DeepSeek API-based teacher model for knowledge distillation"""
    
    def __init__(
        self, 
        api_key: str,
        site_url: str,
        site_name: str,
        model: str = "deepseek/deepseek-r1:free",
        temperature: float = 0.85,
        top_p: float = 1.0,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        repetition_penalty: float = 1,
        top_k: int = 0
    ):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }
        self.model = model
        self.params = {
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k
        }
        
    def get_teacher_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Get response from DeepSeek API"""
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=self.headers,
                data=json.dumps({
                    "model": self.model,
                    "messages": messages,
                    **self.params
                })
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error getting teacher response: {e}")
            return None

class APIKnowledgeDistillation(nn.Module):
    """Knowledge Distillation module using API-based teacher model"""
    
    def __init__(
        self, 
        config,
        teacher_model: APITeacherModel,
        tokenizer,
        hidden_size: int,
        temperature: float = 2.0
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        
        # Projection layers
        self.student_projection = nn.Linear(hidden_size, hidden_size)
        self.knowledge_fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        # Loss functions
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def get_teacher_logits(self, input_texts: List[str]) -> torch.Tensor:
        """Get teacher model logits through API"""
        teacher_responses = []
        for text in input_texts:
            messages = [{"role": "user", "content": text}]
            response = self.teacher_model.get_teacher_response(messages)
            if response:
                teacher_responses.append(response)
            else:
                teacher_responses.append("")  # Empty string as fallback
                
        # Convert responses to logits using tokenizer
        teacher_tokens = self.tokenizer(
            teacher_responses,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        return teacher_tokens['input_ids']
    
    def forward(
        self,
        student_logits: torch.Tensor,
        input_texts: List[str],
        alpha: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for knowledge distillation
        Args:
            student_logits: Logits from student model
            input_texts: Original input texts for teacher model
            alpha: Weight for distillation loss (1-alpha for student loss)
        Returns:
            Dictionary containing losses and combined features
        """
        # Get teacher logits
        with torch.no_grad():
            teacher_logits = self.get_teacher_logits(input_texts)
            
        # Project student features
        student_proj = self.student_projection(student_logits)
        
        # Temperature scaling
        student_temp = F.log_softmax(student_proj / self.temperature, dim=-1)
        teacher_temp = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Calculate losses
        distillation_loss = self.kl_div(student_temp, teacher_temp) * (self.temperature ** 2)
        student_loss = self.mse_loss(student_logits, teacher_logits)
        
        # Combine losses
        total_loss = alpha * distillation_loss + (1 - alpha) * student_loss
        
        # Combine features for knowledge transfer
        combined = torch.cat([student_proj, teacher_logits], dim=-1)
        fused_knowledge = self.knowledge_fusion(combined)
        
        return {
            "loss": total_loss,
            "distillation_loss": distillation_loss,
            "student_loss": student_loss,
            "fused_features": fused_knowledge
        }
