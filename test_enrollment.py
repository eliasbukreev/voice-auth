#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простой тест для проверки работы системы регистрации пользователей
"""

import os
import sys
import glob
from autheticate import SecureVoiceAuthSystem

def test_enrollment():
    """Простая проверка регистрации пользователя"""
    
    print("🎤 ТЕСТ СИСТЕМЫ РЕГИСТРАЦИИ ПОЛЬЗОВАТЕЛЕЙ")
    print("=" * 50)
    
    # 1. Инициализация системы
    try:
        print("\n1️⃣ Инициализация системы...")
        auth_system = SecureVoiceAuthSystem(
            model_path="voice_auth_improved_model.pth"
        )
        print("✅ Система инициализирована успешно!")
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        print("Убедитесь, что файл 'voice_auth_improved_model.pth' существует")
        print("Запустите сначала main.py для обучения модели")
        return False
    
    # 2. Поиск тестовых аудиофайлов
    print("\n2️⃣ Поиск тестовых аудиофайлов...")
    
    # Ищем файлы в директории с обработанными данными
    test_dirs = ["full_data", "processed_data", "data"]
    audio_files = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            files = glob.glob(os.path.join(test_dir, "*.wav"))
            if files:
                audio_files = files[:10]  # Берем первые 10 файлов
                print(f"✅ Найдено {len(files)} аудиофайлов в {test_dir}")
                break
    
    if not audio_files:
        print("❌ Тестовые аудиофайлы не найдены!")
        print("Ожидаемые директории:", test_dirs)
        return False
    
    # 3. Тест регистрации пользователя
    print(f"\n3️⃣ Тест регистрации пользователя...")
    
    test_user = "test_user_001"
    
    # Берем первые 3-5 файлов для регистрации
    enrollment_files = audio_files[:min(5, len(audio_files))]
    print(f"Используем {len(enrollment_files)} файлов для регистрации:")
    for i, file in enumerate(enrollment_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # Пытаемся зарегистрировать пользователя
    try:
        success = auth_system.enroll_user(test_user, enrollment_files)
        
        if success:
            print(f"✅ Пользователь {test_user} успешно зарегистрирован!")
        else:
            print(f"❌ Регистрация пользователя {test_user} не удалась")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка во время регистрации: {e}")
        return False
    
    # 4. Проверка информации о пользователе
    print(f"\n4️⃣ Информация о зарегистрированном пользователе:")
    
    try:
        stats = auth_system.get_user_stats(test_user)
        if stats:
            print(f"  👤 ID пользователя: {stats['user_id']}")
            print(f"  📅 Дата регистрации: {stats['enrolled']}")
            print(f"  🔐 Последняя аутентификация: {stats['last_auth']}")
            print(f"  📊 Размер шаблона: {stats['template_shape']}")
        else:
            print("⚠️ Не удалось получить статистику пользователя")
    except Exception as e:
        print(f"⚠️ Ошибка получения статистики: {e}")
    
    # 5. Список всех пользователей
    print(f"\n5️⃣ Список всех зарегистрированных пользователей:")
    try:
        users = auth_system.list_users()
        print(f"  Всего пользователей: {len(users)}")
        for user in users:
            print(f"  - {user}")
    except Exception as e:
        print(f"⚠️ Ошибка получения списка пользователей: {e}")
    
    # 6. Простой тест аутентификации
    print(f"\n6️⃣ Быстрый тест аутентификации...")
    
    if len(audio_files) > len(enrollment_files):
        # Используем файл, который не участвовал в регистрации
        test_file = audio_files[len(enrollment_files)]
        print(f"Тестируем с файлом: {os.path.basename(test_file)}")
        
        try:
            success, similarity, message = auth_system.authenticate_user(
                test_user, test_file
            )
            
            print(f"  Результат: {'✅ Успех' if success else '❌ Неудача'}")
            print(f"  Сходство: {similarity:.3f}")
            print(f"  Сообщение: {message}")
            
        except Exception as e:
            print(f"⚠️ Ошибка аутентификации: {e}")
    else:
        print("  ⚠️ Недостаточно файлов для теста аутентификации")
    
    print(f"\n7️⃣ Тест защиты от спуфинга...")
    try:
        challenge, token = auth_system.generate_challenge(test_user)
        print(f"  Сгенерированный вызов: '{challenge}'")
        print(f"  Токен: {token[:16]}...")
    except Exception as e:
        print(f"⚠️ Ошибка генерации вызова: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 50)
    
    return True

def cleanup_test_user():
    """Удаляет тестового пользователя после тестирования"""
    try:
        auth_system = SecureVoiceAuthSystem(
            model_path="voice_auth_improved_model.pth"
        )
        
        test_user = "test_user_001"
        if auth_system.delete_user(test_user):
            print(f"🗑️ Тестовый пользователь {test_user} удален")
        
    except Exception as e:
        print(f"⚠️ Ошибка при очистке: {e}")

if __name__ == "__main__":
    print("Запуск теста системы регистрации...\n")
    
    # Проверяем наличие обученной модели
    if not os.path.exists("voice_auth_improved_model.pth"):
        print("❌ Файл модели 'voice_auth_improved_model.pth' не найден!")
        print("Сначала запустите main.py для обучения модели")
        sys.exit(1)
    
    # Запускаем тест
    success = test_enrollment()
    
    if success:
        print("\n" + "="*50)
        print("✅ ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        
        # Спрашиваем, нужно ли очистить тестовые данные
        cleanup = input("\nУдалить тестового пользователя? (y/n): ").lower()
        if cleanup in ['y', 'yes', 'да']:
            cleanup_test_user()
        
    else:
        print("\n" + "="*50)
        print("❌ ТЕСТЫ НЕ ПРОШЛИ")
        print("Проверьте логи выше для диагностики проблем")
    
    print("="*50)