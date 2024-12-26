# Plaka-Tanima-Sistemi
Kritik Açılarda Plaka Bölgesini YOLOv8 ve CNN Kullanarak Bulma ve Karşılaştırma 

Bu çalışmada, zor görüş açılarında araç plakalarının tanınmasını sağlamak amacıyla geliştirilen bir sistem tanıtılmıştır. İlk olarak, farklı açılardan (10°, 20°, 30°, 40°, 50°) plaka görüntüleri toplanmış ve etiketleme işlemleri LabelIMG yazılımı kullanılarak gerçekleştirilmiştir. Daha sonra YOLOv8 algoritması ile model eğitilmiş ve toplam 100 epoch kullanılmıştır. Eğitim sonucunda, plaka bölgesini tespit etme ve karakter okuma performansı değerlendirilmiştir.
Test sonuçları, YOLOv8 algoritmasının, CNN tabanlı yaklaşımlara göre daha yüksek doğruluk oranları sunduğunu göstermiştir. 50°, 40° ve 30° gibi geniş açılarda hem plaka bölgesi tespiti hem de karakter okuma başarıyla gerçekleştirilmiştir. Ancak 20° ve 10° gibi dar açılarda karakter okuma performansı düşmüş, buna karşın plaka bölgesinin tespiti başarılı bir şekilde tamamlanmıştır. Bu durum, açıya bağlı olarak karakter görünürlüğünün azalmasından kaynaklanmıştır.
Çalışma, zor koşullarda bile plaka tanıma sistemlerinin performansını artırmayı hedeflemektedir. Özellikle gerçek uygulamalarda ideal olmayan görüntü koşullarında modelin doğruluğunu değerlendirme ve geliştirme açısından önemli katkılar sunmaktadır.

YOLOV8 ile Plaka Bölgesi bulma Karakter Tanıma Sürecinin Özeti






Veri Hazırlığı:

Etiketleme: Orijinal veri setiniz LabelIMG ile etiketlenerek plakaların konumları belirtilmiştir.

Plaka Bölgesi Tespiti: Sayın  plaka bölgesi başarılı bir şekilde tespit edilmiştir.

Veri Setlerinin Hazırlanması: data.yaml dosyası içerisinde eğitim ve test verileri doğru bir şekilde tanımlanmıştır.
Eğitim Süreci:

YOLOv8 Modeli: YOLOv8 modeli, etiketlenen veriler üzerinde 100 epoch boyunca eğitilmiştir.

Sonraki Adımlar:

Karakter Tanıma: Plaka bölgesi tespit edildikten sonra Semih Ocaklı'nın veri seti kullanılarak karakter tanıma işlemine geçilmiştir.

Model İyileştirme: Gerekli durumlarda performans metriklerini inceleyerek model üzerinde iyileştirmeler yapılabilir.

Performans Metrikleri: 


![image](https://github.com/user-attachments/assets/d6bb1590-b465-4353-aea2-e1dc635f8d3f)


SONUÇLAR:

![image](https://github.com/user-attachments/assets/07a28bf6-6961-4bae-a41a-7f857af94a3f)


