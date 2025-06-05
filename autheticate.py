import numpy as np
import hashlib
import hmac
import secrets
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class VoiceAuthenticator:
    def __init__(self, master_key=None):
        if master_key is None:
            self.master_key = Fernet.generate_key()
        else:
            self.master_key = master_key
        self.cipher = Fernet(self.master_key)
    
    def create_voice_hash(self, voice_features, user_salt=None):
        """Создает безопасный хеш голосовых признаков"""
        if user_salt is None:
            user_salt = secrets.token_bytes(32)
        
        # Преобразуем признаки в байты
        features_bytes = voice_features.numpy().tobytes()
        
        # Создаем хеш с солью
        voice_hash = hashlib.pbkdf2_hmac(
            'sha256', 
            features_bytes, 
            user_salt, 
            100000  # iterations
        )
        
        return voice_hash, user_salt
    
    def encrypt_voice_template(self, voice_features):
        """Шифрует шаблон голоса для хранения"""
        features_bytes = voice_features.numpy().tobytes()
        encrypted = self.cipher.encrypt(features_bytes)
        return encrypted
    
    def decrypt_voice_template(self, encrypted_template):
        """Расшифровывает шаблон голоса"""
        decrypted_bytes = self.cipher.decrypt(encrypted_template)
        # Преобразуем обратно в numpy array (нужно знать исходную форму)
        return np.frombuffer(decrypted_bytes, dtype=np.float32)
    

class SecureVoiceModel:
    def __init__(self, model_path, crypto_key=None):
        self.model = torch.load(model_path)
        self.authenticator = VoiceAuthenticator(crypto_key)
        self.enrolled_users = {}  # user_id -> (encrypted_template, salt)
    
    def enroll_user(self, user_id, voice_features):
        """Регистрирует пользователя"""
        # Создаем хеш для быстрого сравнения
        voice_hash, salt = self.authenticator.create_voice_hash(voice_features)
        
        # Шифруем полный шаблон для точного сравнения
        encrypted_template = self.authenticator.encrypt_voice_template(voice_features)
        
        self.enrolled_users[user_id] = {
            'hash': voice_hash,
            'salt': salt,
            'encrypted_template': encrypted_template,
            'enrollment_time': time.time()
        }
        
        return True
    
    def authenticate_user(self, user_id, voice_features, threshold=0.85):
        """Аутентифицирует пользователя"""
        if user_id not in self.enrolled_users:
            return False, 0.0
        
        user_data = self.enrolled_users[user_id]
        
        # Быстрая проверка по хешу
        test_hash, _ = self.authenticator.create_voice_hash(
            voice_features, user_data['salt']
        )
        
        if test_hash != user_data['hash']:
            return False, 0.0
        
        # Точная проверка через расшифровку и сравнение
        stored_template = self.authenticator.decrypt_voice_template(
            user_data['encrypted_template']
        )
        
        # Вычисляем сходство (cosine similarity)
        similarity = self._calculate_similarity(voice_features.numpy(), stored_template)
        
        return similarity >= threshold, similarity
    
    def _calculate_similarity(self, features1, features2):
        """Вычисляет косинусное сходство"""
        dot_product = np.dot(features1.flatten(), features2.flatten())
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        return dot_product / (norm1 * norm2)
    
class AntiSpoofingAuth:
    def __init__(self):
        self.session_tokens = {}
    
    def generate_challenge(self, user_id):
        """Генерирует случайную фразу для проговаривания"""
        words = ["альфа", "бета", "гамма", "дельта", "эпсилон"]
        challenge = secrets.choice(words) + str(secrets.randbelow(1000))
        
        # Создаем временный токен
        token = secrets.token_urlsafe(32)
        self.session_tokens[token] = {
            'user_id': user_id,
            'challenge': challenge,
            'timestamp': time.time(),
            'used': False
        }
        
        return challenge, token
    
    def verify_challenge_response(self, token, voice_features, transcribed_text):
        """Проверяет ответ на вызов"""
        if token not in self.session_tokens:
            return False
        
        session = self.session_tokens[token]
        
        # Проверяем время (токен действует 60 секунд)
        if time.time() - session['timestamp'] > 60:
            del self.session_tokens[token]
            return False
        
        # Проверяем, что токен не использован
        if session['used']:
            return False
        
        # Проверяем текст
        if transcribed_text.lower() != session['challenge'].lower():
            return False
        
        session['used'] = True
        return True