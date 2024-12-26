# from ultralytics import YOLO
# import os

# # Modeli yükle
# model = YOLO(r"C:\Users\serka\Desktop\license_plate_detector.pt")  # Eğitilmiş modelin yolu

# # Görüntü üzerinde tahmin yap
# results = model('C:/Users/serka/Desktop/yeniplaka/', conf=0.5)

# # Sonuçları kaydetmek için dizin oluştur
# save_dir = 'C:/Users/serka/Desktop/sonuclar/'
# os.makedirs(save_dir, exist_ok=True)  # Eğer klasör yoksa oluştur

# # Sonuçları işleme ve kaydetme
# for i, result in enumerate(results):  # Her bir sonuç için döngü
#     # Görselleştirilmiş sonucu kaydetmek için özel işlem yapılmalı
#     result_plotted = result.plot()  # Tahmin edilen kutucukların çizildiği numpy array

#     # Görüntüyü dosyaya kaydet
#     save_path = os.path.join(save_dir, f"result_{i}.jpg")  # Kaydedilecek dosyanın adı
#     from PIL import Image
#     Image.fromarray(result_plotted).save(save_path)  # Çizilen görüntüyü kaydet
#     print(f"Sonuç kaydedildi: {save_path}")

# print("Tüm sonuçlar kaydedildi.") 


# from ultralytics import YOLO
# import os
# from PIL import Image
# import cv2
# import numpy as np

# # Plaka tespit modeli (birinci model)
# plate_model = YOLO(r"C:\Users\serka\runs\detect\train5\weights\best.pt")

# # Karakter tanıma modeli (ikinci model)
# char_model = YOLO(r"C:\Users\serka\runs\detect\train8\weights\best.pt")

# # Girdi görüntülerinin bulunduğu klasör
# input_dir = 'C:/Users/serka/Desktop/yeniplaka/'
# output_dir = 'C:/Users/serka/Desktop/sonuclar/'
# os.makedirs(output_dir, exist_ok=True)  # Çıktılar için klasör oluştur

# # Girdi görüntülerini oku
# image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# for img_path in image_paths:
#     # Plaka tespiti yap
#     plate_results = plate_model(img_path, conf=0.5)
#     img = cv2.imread(img_path)
    
#     for i, plate_result in enumerate(plate_results):
#         # Tespit edilen plaka kutularını işle
#         for bbox in plate_result.boxes.xyxy:  # x_min, y_min, x_max, y_max
#             x_min, y_min, x_max, y_max = map(int, bbox)

#             # Plaka bölgesini kırp
#             cropped_plate = img[y_min:y_max, x_min:x_max]

#             # Karakter tanıma modeli ile plaka üzerindeki karakterleri tespit et
#             if cropped_plate is not None and cropped_plate.size > 0:
#                 cropped_plate_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
#                 char_results = char_model(cropped_plate_rgb, conf=0.5)

#                 # Karakter sonuçlarını birleştir
#                 recognized_text = ''
#                 for char_result in char_results:
#                     for char_bbox, char_cls in zip(char_result.boxes.xyxy, char_result.boxes.cls):
#                         recognized_text += char_model.names[int(char_cls)]

#                 # Sonuçları görselleştir ve kaydet
#                 result_path = os.path.join(output_dir, f"result_{os.path.basename(img_path)}_{i}.jpg")
#                 Image.fromarray(cropped_plate_rgb).save(result_path)

#                 print(f"Plaka: {recognized_text} - Kaydedildi: {result_path}")

# print("Tüm işlem tamamlandı.")









# from ultralytics import YOLO
# import os
# from PIL import Image
# import cv2
# import numpy as np

# # Plaka tespit modeli (birinci model)
# plate_model = YOLO(r"C:\Users\serka\runs\detect\train5\weights\best.pt")

# # Karakter tanıma modeli (ikinci model)
# char_model = YOLO(r"C:\Users\serka\runs\detect\train9\weights\best.pt")

# # Girdi görüntülerinin bulunduğu klasör
# input_dir = 'C:/Users/serka/Desktop/iloveimg-converted/iloveimg-converted/30/'
# output_dir = 'C:/Users/serka/Desktop/sonuclar30drc/'
# os.makedirs(output_dir, exist_ok=True)  # Çıktılar için klasör oluştur

# # Girdi görüntülerini oku
# image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# for img_path in image_paths:
#     # Plaka tespiti yap
#     plate_results = plate_model(img_path, conf=0.5)
#     img = cv2.imread(img_path)
    
#     for i, plate_result in enumerate(plate_results):
#         # Tespit edilen plaka kutularını işle
#         for bbox in plate_result.boxes.xyxy:  # x_min, y_min, x_max, y_max
#             x_min, y_min, x_max, y_max = map(int, bbox)

#             # Plaka bölgesini kırp
#             cropped_plate = img[y_min:y_max, x_min:x_max]

#             # Karakter tanıma modeli ile plaka üzerindeki karakterleri tespit et
#             if cropped_plate is not None and cropped_plate.size > 0:
#                 cropped_plate_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
#                 char_results = char_model(cropped_plate_rgb, conf=0.5)

#                 # Karakter sonuçlarını birleştir ve soldan sağa sırala
#                 recognized_text = ''
#                 char_bboxes_and_classes = []
#                 for char_result in char_results:
#                     for char_bbox, char_cls in zip(char_result.boxes.xyxy, char_result.boxes.cls):
#                         char_bboxes_and_classes.append((char_bbox[0], char_cls))  # x_min ve sınıf

#                 char_bboxes_and_classes.sort(key=lambda x: x[0])  # Soldan sağa sırala
#                 recognized_text = ''.join([char_model.names[int(cls)] for _, cls in char_bboxes_and_classes])

#                 # Sonuçları görselleştir ve kaydet
#                 result_image = cv2.putText(
#                     img.copy(),
#                     f"Plate: {recognized_text}",
#                     (x_min, y_min - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9,
#                     (0, 255, 0),
#                     2
#                 )
#                 result_image = cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#                 result_path = os.path.join(output_dir, f"result_{recognized_text}_{i}.jpg")
#                 Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).save(result_path)

#                 print(f"Plaka: {recognized_text} - Kaydedildi: {result_path}")

# print("Tüm işlem tamamlandı.")

# ***********************************************************************************************************DNEME 2




































# from ultralytics import YOLO
# import os
# from PIL import Image
# import cv2
# import numpy as np
# import re  # Türkiye plaka formatı için

# # Türkiye plaka karakter seti
# VALID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# # Türkiye plaka formatını kontrol eden bir fonksiyon
# def validate_turkish_plate(recognized_text):
#     # Geçerli karakterlerden oluşmayanları kaldır
#     recognized_text = ''.join([c for c in recognized_text if c in VALID_CHARS])

#     # Plakayı üç bloğa ayırmak için regex
#     pattern = re.compile(r'^(\d{2})\s?([A-Z]{1,3})\s?(\d{1,4})$')
#     match = pattern.match(recognized_text)

#     if match:
#         return f"{match.group(1)} {match.group(2)} {match.group(3)}"
#     return recognized_text  # Format uygun değilse olduğu gibi döndür

# # Plaka tespit modeli (birinci model)
# plate_model = YOLO(r"C:\Users\serka\runs\detect\train5\weights\best.pt")

# # Karakter tanıma modeli (ikinci model)
# char_model = YOLO(r"C:\Users\serka\runs\detect\train9\weights\best.pt")

# # Girdi görüntülerinin bulunduğu klasör
# input_dir = r"C:\Users\serka\Desktop\Valilik-Makam-Arac-Plasi.jpg"
# output_dir = 'C:/Users/serka/Desktop/sonuclarNilgünHocanınArabası/'
# os.makedirs(output_dir, exist_ok=True)  # Çıktılar için klasör oluştur

# # Girdi görüntülerini oku
# image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# for img_path in image_paths:
#     # Plaka tespiti yap
#     plate_results = plate_model(img_path, conf=0.5)
#     img = cv2.imread(img_path)

#     for i, plate_result in enumerate(plate_results):
#         # Tespit edilen plaka kutularını işle
#         for bbox in plate_result.boxes.xyxy:  # x_min, y_min, x_max, y_max
#             x_min, y_min, x_max, y_max = map(int, bbox)

#             # Plaka bölgesini kırp
#             cropped_plate = img[y_min:y_max, x_min:x_max]

#             # Karakter tanıma modeli ile plaka üzerindeki karakterleri tespit et
#             if cropped_plate is not None and cropped_plate.size > 0:
#                 cropped_plate_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
#                 char_results = char_model(cropped_plate_rgb, conf=0.5)

#                 # Karakter sonuçlarını birleştir ve soldan sağa sırala
#                 recognized_text = ''
#                 char_bboxes_and_classes = []
#                 for char_result in char_results:
#                     for char_bbox, char_cls in zip(char_result.boxes.xyxy, char_result.boxes.cls):
#                         char_bboxes_and_classes.append((char_bbox[0], char_cls))  # x_min ve sınıf

#                 char_bboxes_and_classes.sort(key=lambda x: x[0])  # Soldan sağa sırala
#                 recognized_text = ''.join([char_model.names[int(cls)] for _, cls in char_bboxes_and_classes])

#                 # Türkiye plaka formatına göre düzeltme
#                 corrected_plate = validate_turkish_plate(recognized_text)

#                 # Sonuçları görselleştir ve kaydet
#                 result_image = cv2.putText(
#                     img.copy(),
#                     f"Plate: {corrected_plate}",
#                     (x_min, y_min - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.9,
#                     (0, 255, 0),
#                     2
#                 )
#                 result_image = cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#                 result_path = os.path.join(output_dir, f"result_{corrected_plate}_{i}.jpg")
#                 Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).save(result_path)

#                 print(f"Plaka: {corrected_plate} - Kaydedildi: {result_path}")

# print("Tüm işlem tamamlandı.")



# ****************************************
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np
import re  # Türkiye plaka formatı için
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import csv

# Türkiye plaka karakter seti
VALID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Türkiye plaka formatını kontrol eden bir fonksiyon
def validate_turkish_plate(recognized_text):
    # Geçerli karakterlerden oluşmayanları kaldır
    recognized_text = ''.join([c for c in recognized_text if c in VALID_CHARS])

    # Plakayı üç bloğa ayırmak için regex
    pattern = re.compile(r'^(\d{2})\s?([A-Z]{1,3})\s?(\d{1,4})$')
    match = pattern.match(recognized_text)

    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3)}"
    return recognized_text  # Format uygun değilse olduğu gibi döndür

# Plaka tespit modeli (birinci model)
plate_model = YOLO(r"C:\\Users\\serka\\runs\\detect\\train5\\weights\\best.pt")

# Karakter tanıma modeli (ikinci model)
char_model = YOLO(r"C:\\Users\\serka\\runs\\detect\\train9\\weights\\best.pt")

# Ana veri klasörü
base_input_dir = 'C:/Users/serka/Desktop/TEST-VERISETI/'
output_base_dir = 'C:/Users/serka/Desktop/sonuclar/'
os.makedirs(output_base_dir, exist_ok=True)

# Açılar için klasör adları
angle_folders = ['10', '20', '30', '40', '50']

# Performans metriklerini saklamak için bir liste (CSV yazmadan önce toplanacak)
performance_metrics = []

for angle in angle_folders:
    input_dir = os.path.join(base_input_dir, angle)
    output_dir = os.path.join(output_base_dir, f"sonuclar_{angle}drc/")
    os.makedirs(output_dir, exist_ok=True)  # Çıktılar için klasör oluştur

    # Girdi görüntülerini oku
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    true_labels = []
    predicted_labels = []

    for img_path in image_paths:
        # Gerçek etiket dosyasını al (aynı isimli label.csv dosyasını bekliyoruz)
        label_path = os.path.join(input_dir, 'label.csv')
        if not os.path.exists(label_path):  # Dosyanın var olup olmadığını kontrol et
            print(f"{label_path} bulunamadı. Bu klasör atlanacak.")
            continue  # Eğer dosya yoksa, bu klasörü atla

        with open(label_path, 'r') as label_file:
            label_reader = csv.reader(label_file)
            true_labels_for_image = {}
            for row in label_reader:
                if len(row) >= 2:  # Satırda en az iki öğe olup olmadığını kontrol et
                    true_labels_for_image[row[0]] = row[1]
                else:
                    print(f"Geçersiz satır (Eksik veri): {row}")

        true_label = true_labels_for_image.get(os.path.basename(img_path), "")
        true_labels.append(true_label)

        # Plaka tespiti yap
        plate_results = plate_model(img_path, conf=0.5)
        img = cv2.imread(img_path)

        recognized_text = ""

        for i, plate_result in enumerate(plate_results):
            # Tespit edilen plaka kutularını işle
            for bbox in plate_result.boxes.xyxy:  # x_min, y_min, x_max, y_max
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Plaka bölgesini kırp
                cropped_plate = img[y_min:y_max, x_min:x_max]

                # Karakter tanıma modeli ile plaka üzerindeki karakterleri tespit et
                if cropped_plate is not None and cropped_plate.size > 0:
                    cropped_plate_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
                    char_results = char_model(cropped_plate_rgb, conf=0.5)

                    # Karakter sonuçlarını birleştir ve soldan sağa sırala
                    char_bboxes_and_classes = []
                    for char_result in char_results:
                        for char_bbox, char_cls in zip(char_result.boxes.xyxy, char_result.boxes.cls):
                            char_bboxes_and_classes.append((char_bbox[0], char_cls))  # x_min ve sınıf

                    char_bboxes_and_classes.sort(key=lambda x: x[0])  # Soldan sağa sırala
                    recognized_text = ''.join([char_model.names[int(cls)] for _, cls in char_bboxes_and_classes])

                    # Türkiye plaka formatına göre düzeltme
                    recognized_text = validate_turkish_plate(recognized_text)

                    # Sonuçları görselleştir ve kaydet
                    result_image = cv2.putText(
                        img.copy(),
                        f"Plate: {recognized_text}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
                    result_image = cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    result_path = os.path.join(output_dir, f"result_{recognized_text}_{i}.jpg")
                    Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)).save(result_path)

                    print(f"Plaka: {recognized_text} - Kaydedildi: {result_path}")

        predicted_labels.append(recognized_text)

    # Performans metriklerini hesapla
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    mcc = matthews_corrcoef(true_labels, predicted_labels)

    # Sonuçları listeye ekle
    performance_metrics.append([angle, accuracy, f1, precision, recall, mcc])

# Tüm performans metriklerini konsola düzgün bir formatta yazdır
print("\nTüm performans metrikleri:")
print(f"{'Angle':<10} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'MCC':<10}")
for metrics in performance_metrics:
    print(f"{metrics[0]:<10} {metrics[1]:<10.4f} {metrics[2]:<10.4f} {metrics[3]:<10.4f} {metrics[4]:<10.4f} {metrics[5]:<10.4f}")

# Performans metriklerini CSV dosyasına yaz
performance_file = os.path.join(output_base_dir, 'performance_metrics.csv')
with open(performance_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Angle", "Accuracy", "F1 Score", "Precision", "Recall", "MCC"])
    for metrics in performance_metrics:
        writer.writerow(metrics)

print("\nTüm işlem tamamlandı.")
