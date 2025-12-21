# 🧪 PAPAGAN RAG - Test Rehberi

## Hızlı Başlangıç

### Testleri Çalıştır
```bash
# Tüm testler
pytest test_simple.py -v

# Belirli test kategorisi
pytest test_simple.py::TestBasic -v
pytest test_simple.py::TestSecurity -v

# Kod kapsamı
pytest test_simple.py --cov=main
```

## Test Kategorileri

### ✅ Birim Testler (3 test)
- VectorStore yükleme
- Dosya doğrulama
- Giriş temizlemesi

### ⚠️ Hata İşleme (2 test)
- Geçersiz PDF işlemesi
- None vectorstore kontrollü

### 🔐 Güvenlik (2 test)
- SQL injection saldırı bloklama
- Dosya yolu traversal saldırı bloklama

### 🔗 Entegrasyon (1 test)
- Dokuman parçalama

## Test Sonuçları

```
8/8 test geçmeli
Başarı oranı: 100%
```

## Komut Satırı Örnekleri

```bash
# Verbose çıktı
pytest test_simple.py -v

# Hızlı test
pytest test_simple.py -q

# Belirli test
pytest test_simple.py::TestBasic::test_file_validation -v

# Kod kapsamı raporu
pytest test_simple.py --cov=main --cov-report=html
```

---

**Durum:** ✅ HAZIR  
**Son Güncelleme:** 2025-12-20
