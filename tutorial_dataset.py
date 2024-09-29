import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_path):
        """
        Параметры:
        - data_path: путь к папке, в которой находятся данные
        """
        self.data = []
        self.data_path = data_path  # Сохраняем переданный путь
        with open(f'{self.data_path}/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        # Формируем полные пути к файлам
        source = cv2.imread(f'{self.data_path}/{source_filename}')
        target = cv2.imread(f'{self.data_path}/{target_filename}')

        # Не забываем, что OpenCV читает изображения в формате BGR.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Нормализуем изображение source в диапазон [0, 1].
        source = source.astype(np.float32) / 255.0

        # Нормализуем изображение target в диапазон [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

# Пример использования
# dataset = MyDataset(data_path='./training/fill50k')
