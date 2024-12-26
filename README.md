# Plaka-Tanima-Sistemi
Kritik Açılarda Plaka Bölgesini YOLOv8 ve CNN Kullanarak Bulma ve Karşılaştırma 

Bu çalışmada, zor görüş açılarında araç plakalarının tanınmasını sağlamak amacıyla geliştirilen bir sistem tanıtılmıştır. İlk olarak, farklı açılardan (10°, 20°, 30°, 40°, 50°) plaka görüntüleri toplanmış ve etiketleme işlemleri LabelIMG yazılımı kullanılarak gerçekleştirilmiştir. Daha sonra YOLOv8 algoritması ile model eğitilmiş ve toplam 100 epoch kullanılmıştır. Eğitim sonucunda, plaka bölgesini tespit etme ve karakter okuma performansı değerlendirilmiştir.
Test sonuçları, YOLOv8 algoritmasının, CNN tabanlı yaklaşımlara göre daha yüksek doğruluk oranları sunduğunu göstermiştir. 50°, 40° ve 30° gibi geniş açılarda hem plaka bölgesi tespiti hem de karakter okuma başarıyla gerçekleştirilmiştir. Ancak 20° ve 10° gibi dar açılarda karakter okuma performansı düşmüş, buna karşın plaka bölgesinin tespiti başarılı bir şekilde tamamlanmıştır. Bu durum, açıya bağlı olarak karakter görünürlüğünün azalmasından kaynaklanmıştır.
Çalışma, zor koşullarda bile plaka tanıma sistemlerinin performansını artırmayı hedeflemektedir. Özellikle gerçek uygulamalarda ideal olmayan görüntü koşullarında modelin doğruluğunu değerlendirme ve geliştirme açısından önemli katkılar sunmaktadır.

