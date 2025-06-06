import os
import torch
import numpy as np
import librosa
import hashlib
import hmac
import secrets
import time
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from collections import defaultdict

# Импорт ваших модулей
from model import VoiceAuthCNN
from dataset import VoiceDataset
from evaluate import evaluate_model

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
        if isinstance(voice_features, torch.Tensor):
            features_bytes = voice_features.cpu().numpy().tobytes()
        else:
            features_bytes = voice_features.tobytes()
        
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
        if isinstance(voice_features, torch.Tensor):
            features_bytes = voice_features.cpu().numpy().tobytes()
        else:
            features_bytes = voice_features.tobytes()
            
        encrypted = self.cipher.encrypt(features_bytes)
        return encrypted
    
    def decrypt_voice_template(self, encrypted_template, shape):
        """Расшифровывает шаблон голоса"""
        decrypted_bytes = self.cipher.decrypt(encrypted_template)
        # Преобразуем обратно в numpy array
        features = np.frombuffer(decrypted_bytes, dtype=np.float32)
        return features.reshape(shape)


class SecureVoiceAuthSystem:
    def __init__(self, model_path="voice_auth_improved_model.pth", crypto_key=None, db_path="secure_voice_db.json"):
        """Инициализация безопасной системы голосовой аутентификации"""
        print("🔐 Initializing Secure Voice Authentication System...")
        
        # Загружаем обученную модель
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_data = torch.load(model_path, map_location=self.device)
        
        # Восстанавливаем модель
        config = self.model_data['config']
        self.model = VoiceAuthCNN(
            input_features=config['n_mfcc'],
            output_dim=config['output_dim']
        ).to(self.device)
        self.model.load_state_dict(self.model_data['model_state_dict'])
        self.model.eval()
        
        # Конфигурация из обученной модели
        self.n_mfcc = config['n_mfcc']
        self.max_len = config['max_len']
        self.sr = 16000
        
        # Система шифрования
        self.authenticator = VoiceAuthenticator(crypto_key)
        
        # База данных пользователей
        self.db_path = db_path
        self.enrolled_users = self._load_database()
        
        # Система защиты от спуфинга
        self.anti_spoofing = AntiSpoofingAuth()
        
        # Пороги безопасности
        self.auth_threshold = 0.75  # Можно настроить на основе EER
        self.enrollment_threshold = 0.8
        
        print(f"✅ System initialized on {self.device}")
        print(f"📊 Model performance: EER={self.model_data['results']['eer']:.4f}")
        print(f"👥 Enrolled users: {len(self.enrolled_users)}")
    
    def _load_database(self):
        """Загружает базу данных пользователей"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    # Декодируем base64 данные
                    for user_id, user_data in data.items():
                        user_data['hash'] = base64.b64decode(user_data['hash'])
                        user_data['salt'] = base64.b64decode(user_data['salt'])
                        user_data['encrypted_template'] = base64.b64decode(user_data['encrypted_template'])
                    return data
            except Exception as e:
                print(f"⚠️ Error loading database: {e}")
                return {}
        return {}
    
    def _save_database(self):
        """Сохраняет базу данных пользователей"""
        try:
            # Кодируем binary данные в base64 для JSON
            encoded_data = {}
            for user_id, user_data in self.enrolled_users.items():
                encoded_data[user_id] = {
                    'hash': base64.b64encode(user_data['hash']).decode('utf-8'),
                    'salt': base64.b64encode(user_data['salt']).decode('utf-8'),
                    'encrypted_template': base64.b64encode(user_data['encrypted_template']).decode('utf-8'),
                    'template_shape': user_data['template_shape'],
                    'enrollment_time': user_data['enrollment_time'],
                    'last_auth': user_data.get('last_auth', 0)
                }
            
            with open(self.db_path, 'w') as f:
                json.dump(encoded_data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving database: {e}")
    
    def _extract_voice_features(self, audio_path):
        """Извлекает признаки из аудиофайла"""
        try:
            # Загружаем аудио
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Проверяем минимальную длину
            if len(y) < self.sr * 0.5:
                min_len = int(self.sr * 0.5)
                if len(y) > 0:
                    y = np.tile(y, (min_len // len(y)) + 1)[:min_len]
                else:
                    return None
            
            # Извлекаем признаки (аналогично dataset.py)
            mfcc = librosa.feature.mfcc(
                y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                n_fft=1024, hop_length=256, win_length=1024
            )
            
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            spectral_centroids = librosa.feature.spectral_centroid(
                y=y, sr=self.sr, hop_length=256
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=y, sr=self.sr, hop_length=256
            )
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                y, hop_length=256
            )
            
            chroma = librosa.feature.chroma_stft(
                y=y, sr=self.sr, hop_length=256, n_fft=1024
            )
            
            # Объединяем признаки
            features = np.vstack([
                mfcc, delta_mfcc, delta2_mfcc,
                spectral_centroids, spectral_rolloff,
                zero_crossing_rate, chroma
            ])
            
            # Обрабатываем длину
            if features.shape[1] > self.max_len:
                start = (features.shape[1] - self.max_len) // 2
                features = features[:, start:start + self.max_len]
            else:
                pad_width = self.max_len - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)),
                                mode='constant', constant_values=0)
            
            # Нормализация
            features = (features - np.mean(features, axis=1, keepdims=True))
            std = np.std(features, axis=1, keepdims=True)
            features = features / (std + 1e-8)
            features = np.clip(features, -3, 3)
            
            return torch.tensor(features).float().unsqueeze(0).to(self.device)
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None
    
    def _get_voice_embedding(self, audio_path):
        """Получает эмбеддинг голоса из аудиофайла"""
        features = self._extract_voice_features(audio_path)
        if features is None:
            return None
        
        with torch.no_grad():
            embedding = self.model(features)
            return embedding.squeeze(0)  # Убираем batch dimension
    
    def enroll_user(self, user_id, audio_samples):
        """
        Регистрирует пользователя на основе нескольких аудиосэмплов
        audio_samples: список путей к аудиофайлам пользователя
        """
        print(f"👤 Enrolling user: {user_id}")
        
        if len(audio_samples) < 3:
            print("❌ Need at least 3 audio samples for enrollment")
            return False
        
        embeddings = []
        for audio_path in audio_samples:
            embedding = self._get_voice_embedding(audio_path)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) < 2:
            print("❌ Not enough valid audio samples")
            return False
        
        # Создаем шаблон пользователя (медиана для устойчивости)
        embeddings_tensor = torch.stack(embeddings)
        template = torch.median(embeddings_tensor, dim=0)[0]
        
        # Проверяем качество шаблона
        similarities = []
        for emb in embeddings:
            sim = torch.cosine_similarity(template.unsqueeze(0), emb.unsqueeze(0))
            similarities.append(sim.item())
        
        avg_similarity = np.mean(similarities)
        if avg_similarity < self.enrollment_threshold:
            print(f"❌ Template quality too low: {avg_similarity:.3f}")
            return False
        
        # Создаем безопасный хеш и шифруем шаблон
        voice_hash, salt = self.authenticator.create_voice_hash(template)
        encrypted_template = self.authenticator.encrypt_voice_template(template)
        
        # Сохраняем данные пользователя
        self.enrolled_users[user_id] = {
            'hash': voice_hash,
            'salt': salt,
            'encrypted_template': encrypted_template,
            'template_shape': template.shape,
            'enrollment_time': time.time(),
            'last_auth': 0
        }
        
        self._save_database()
        print(f"✅ User {user_id} enrolled successfully (quality: {avg_similarity:.3f})")
        return True
    
    def authenticate_user(self, user_id, audio_path, use_anti_spoofing=True):
        """
        Аутентифицирует пользователя
        """
        print(f"🔍 Authenticating user: {user_id}")
        
        if user_id not in self.enrolled_users:
            print("❌ User not enrolled")
            return False, 0.0, "User not found"
        
        # Получаем эмбеддинг из аудио
        test_embedding = self._get_voice_embedding(audio_path)
        if test_embedding is None:
            print("❌ Could not extract features from audio")
            return False, 0.0, "Invalid audio"
        
        user_data = self.enrolled_users[user_id]
        
        # Быстрая проверка по хешу
        test_hash, _ = self.authenticator.create_voice_hash(
            test_embedding, user_data['salt']
        )
        
        # Точная проверка через расшифровку шаблона
        stored_template = self.authenticator.decrypt_voice_template(
            user_data['encrypted_template'],
            user_data['template_shape']
        )
        stored_template = torch.tensor(stored_template).to(self.device)
        
        # Вычисляем косинусное сходство
        similarity = torch.cosine_similarity(
            test_embedding.unsqueeze(0),
            stored_template.unsqueeze(0)
        ).item()
        
        # Проверяем порог
        is_authenticated = similarity >= self.auth_threshold
        
        if is_authenticated:
            # Обновляем время последней аутентификации
            self.enrolled_users[user_id]['last_auth'] = time.time()
            self._save_database()
            print(f"✅ Authentication successful (similarity: {similarity:.3f})")
            return True, similarity, "Success"
        else:
            print(f"❌ Authentication failed (similarity: {similarity:.3f})")
            return False, similarity, "Similarity too low"
    
    def generate_challenge(self, user_id):
        """Генерирует вызов для защиты от спуфинга"""
        return self.anti_spoofing.generate_challenge(user_id)
    
    def verify_challenge_response(self, token, audio_path, transcribed_text):
        """Проверяет ответ на вызов"""
        # Сначала проверяем текст
        text_valid = self.anti_spoofing.verify_challenge_response(
            token, None, transcribed_text
        )
        
        if not text_valid:
            return False, "Challenge failed"
        
        # Затем можно добавить проверку голоса
        return True, "Challenge passed"
    
    def get_user_stats(self, user_id):
        """Получает статистику пользователя"""
        if user_id not in self.enrolled_users:
            return None
        
        user_data = self.enrolled_users[user_id]
        return {
            'user_id': user_id,
            'enrolled': time.ctime(user_data['enrollment_time']),
            'last_auth': time.ctime(user_data['last_auth']) if user_data['last_auth'] > 0 else "Never",
            'template_shape': user_data['template_shape']
        }
    
    def list_users(self):
        """Список всех зарегистрированных пользователей"""
        return list(self.enrolled_users.keys())
    
    def delete_user(self, user_id):
        """Удаляет пользователя из системы"""
        if user_id in self.enrolled_users:
            del self.enrolled_users[user_id]
            self._save_database()
            print(f"🗑️ User {user_id} deleted")
            return True
        return False


class AntiSpoofingAuth:
    def __init__(self):
        self.session_tokens = {}
    
    def generate_challenge(self, user_id):
        """Генерирует случайную фразу для проговаривания"""
        words = ["альфа", "бета", "гамма", "дельта", "эпсилон", "зета", "тета"]
        numbers = [str(secrets.randbelow(100)) for _ in range(2)]
        challenge = " ".join(secrets.sample(words, 2) + numbers)
        
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
        if transcribed_text.lower().strip() != session['challenge'].lower().strip():
            return False
        
        session['used'] = True
        return True


# Пример использования системы
def demo_secure_voice_auth():
    """Демонстрация работы системы"""
    print("🎤 SECURE VOICE AUTHENTICATION SYSTEM DEMO")
    print("=" * 50)
    
    # Инициализация системы
    try:
        auth_system = SecureVoiceAuthSystem(
            model_path="voice_auth_improved_model.pth"
        )
    except Exception as e:
        print(f"❌ Could not initialize system: {e}")
        print("Make sure you have run main.py to train the model first!")
        return
    
    # Пример регистрации пользователя
    print("\n1. ENROLLMENT DEMO")
    user_id = "demo_user_001"
    
    # В реальном использовании здесь были бы пути к аудиофайлам пользователя
    sample_files = [
        "path/to/user/sample1.wav",
        "path/to/user/sample2.wav",
        "path/to/user/sample3.wav"
    ]
    
    print(f"Enrolling user: {user_id}")
    # enrollment_result = auth_system.enroll_user(user_id, sample_files)
    
    # Пример аутентификации
    print("\n2. AUTHENTICATION DEMO")
    test_audio = "path/to/test/audio.wav"
    # auth_result, similarity, message = auth_system.authenticate_user(user_id, test_audio)
    
    # Пример с защитой от спуфинга
    print("\n3. ANTI-SPOOFING DEMO")
    challenge, token = auth_system.generate_challenge(user_id)
    print(f"Challenge: {challenge}")
    print(f"Token: {token[:16]}...")
    
    # Статистика пользователей
    print("\n4. USER MANAGEMENT")
    users = auth_system.list_users()
    print(f"Enrolled users: {users}")
    
    for user in users:
        stats = auth_system.get_user_stats(user)
        if stats:
            print(f"User {user}: enrolled {stats['enrolled']}, last auth {stats['last_auth']}")


if __name__ == "__main__":
    demo_secure_voice_auth()
