import os
import csv
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import random


def convert_common_voice_to_wav(tsv_path, input_dir, output_dir, target_sr=16000):
    """
    Конвертирует все файлы из Common Voice в WAV формат
    """
    print(f"\n{'=' * 80}")
    print(f"Starting full database conversion")
    print(f"TSV path: {tsv_path}")
    print(f"Audio dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'=' * 80}\n")

    # Проверка существования путей
    if not os.path.exists(tsv_path):
        print(f"❌ TSV file not found: {tsv_path}")
        return 0

    if not os.path.exists(input_dir):
        print(f"❌ Audio directory not found: {input_dir}")
        return 0

    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)

    # Проверка структуры TSV
    print("Validating TSV structure...")
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            headers = f.readline().strip().split('\t')
            if 'client_id' not in headers or 'path' not in headers:
                print("❌ Required columns (client_id, path) not found in TSV")
                return 0
            print(f"TSV contains {len(headers)} columns")
    except Exception as e:
        print(f"❌ Error reading TSV: {str(e)}")
        return 0

    # Счетчики
    user_counter = {}
    converted_count = 0
    skipped_count = 0

    try:
        # Первый проход: подсчет общего количества строк
        with open(tsv_path, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1  # Минус заголовок

        # Второй проход: обработка данных
        with open(tsv_path, 'r', encoding='utf-8') as f:
            # Пропускаем заголовок
            f.readline()

            for row in tqdm(f, total=total_rows, desc="Converting audio"):
                try:
                    parts = row.strip().split('\t')
                    if len(parts) < 2:
                        skipped_count += 1
                        continue

                    client_id = parts[0]
                    file_path = parts[1]

                    # Формируем путь к файлу
                    input_path = os.path.join(input_dir, file_path)

                    # Проверяем существование файла
                    if not os.path.exists(input_path):
                        # Пробуем разные расширения
                        for ext in ['.mp3', '.ogg', '.wav', '.flac', '.m4a']:
                            test_path = f"{input_path}{ext}"
                            if os.path.exists(test_path):
                                input_path = test_path
                                break
                        else:
                            skipped_count += 1
                            continue

                    # Инициализируем счетчик для пользователя
                    if client_id not in user_counter:
                        user_counter[client_id] = 0
                    user_counter[client_id] += 1

                    # Формируем выходной путь
                    output_path = os.path.join(
                        output_dir,
                        f"user_{client_id}_{user_counter[client_id]:05d}.wav"
                    )

                    # Пропускаем уже сконвертированные файлы
                    if os.path.exists(output_path):
                        converted_count += 1
                        continue

                    # Конвертация
                    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
                    y = librosa.util.normalize(y)
                    sf.write(output_path, y, target_sr, subtype='PCM_16')

                    converted_count += 1
                except Exception as e:
                    skipped_count += 1
                    continue

    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        return 0

    print(f"\n{'=' * 80}")
    print(f"✅ Conversion complete!")
    print(f"Total users: {len(user_counter)}")
    print(f"Converted files: {converted_count}")
    print(f"Skipped files: {skipped_count}")
    print(f"Total rows processed: {total_rows}")
    print(f"{'=' * 80}")

    return converted_count