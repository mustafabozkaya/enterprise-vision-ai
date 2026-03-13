# Streamlit UI + Gradio HF Space Redesign

> **Date:** 2026-03-13
> **Status:** Approved
> **Scope:** Full rebuild of Streamlit UI (demo/sunum odaklı) + Gradio HF Space (hafif demo + API)

---

## 1. Hedef

Mevcut Streamlit UI ve Gradio uygulamalarını sıfırdan yeniden yaz. Streamlit: dahili demo/sunum için profesyonel, temiz, bilgi odaklı ana sayfa + üst/alt layout'lu modül sayfaları. Gradio: HuggingFace Space için hafif, tek sayfada iki modül, API endpoint aktif.

---

## 2. Yaklaşım

**Minimal Rebuild (Yaklaşım 1)** seçildi:
- Fazladan abstraction yok, component library yok
- `app.py` + `pages/defect.py` + `pages/ore.py` sıfırdan yazılır
- `app.py` kendi `render_sidebar()` tanımlamak yerine `from pages._sidebar import render_sidebar` kullanır
- `pages/_sidebar.py` mevcut haliyle korunur, değişiklik yapılmaz
- Inference helpers için `enterprise_vision_ai.utils` modüllerinden (image_utils, metrics, visualization) doğrudan import yapılır — `services/utils.py` DeprecationWarning içerdiğinden KULLANILMAZ
- Gradio: `huggingface_space/app.py` sıfırdan, `gr.Blocks` + `gr.Tabs` (mevcut broken `gr.TabbedInterface` nesting tamamen kaldırılır)

---

## 3. Streamlit Dosya Yapısı

```
app.py                          # Ana sayfa — bilgi/showcase sayfası (sıfırdan yeniden yazılır)
pages/
  _sidebar.py                   # Shared sidebar — DOKUNULMAZ
  01_Defekt_Tespiti.py          # Wrapper — değişmez
  02_Cevher_On_Secimi.py        # Wrapper — değişmez
  defect.py                     # Sıfırdan yeniden yazılır
  ore.py                        # Sıfırdan yeniden yazılır
huggingface_space/
  app.py                        # Sıfırdan yeniden yazılır
  examples/                     # Oluşturulur: 2 örnek görüntü (defect_sample.jpg, ore_sample.jpg)
```

---

## 4. Streamlit Sayfa Tasarımları

### 4.1 Ana Sayfa (`app.py`)

**Önemli:** Mevcut `app.py`'deki yerel `render_sidebar()` fonksiyonu kaldırılır. Yerine:
```python
from pages._sidebar import render_sidebar
render_sidebar(active_page="")
```

**Bölümler (yukarıdan aşağı):**

1. **Hero** — `st.title` + alt başlık, tek satır açıklama
2. **Metrik satırı** — 4 `st.metric` kutusu: Aktif Modüller (2), Model Doğruluğu (%94), İşlem Hızı (30 FPS), Desteklenen Sınıf (9+)
3. **Modül kartları** — 2 sütun, her biri: ikon + başlık + özellik listesi (`<ul>`) + `st.page_link` butonu. Sol kart: Defekt Tespiti. Sağ kart: Cevher Ön Seçimi.
4. **Nasıl Kullanılır** — `st.expander` ile Defekt ve Cevher modülü için adım adım talimatlar
5. **Footer** — copyright

---

### 4.2 Defekt Tespiti (`pages/defect.py`)

**Önemli:** Mevcut otomatik çalışma (upload → anında inference) kaldırılır. Yeni akış: upload + button click → inference, `st.session_state` kullanılır.

**Import:**
```python
from enterprise_vision_ai.utils.image_utils import load_image, preprocess_for_model
from enterprise_vision_ai.utils.visualization import draw_annotations
from enterprise_vision_ai.utils.metrics import create_metrics_dataframe
```

**Layout (üst → alt):**

1. **Başlık bloku** — `st.title("🔍 Defekt Tespiti")` + açıklama satırı
2. **Yükleme alanı** — `st.file_uploader` (JPG, JPEG, PNG) tam genişlik
3. **"Analiz Et" butonu** — primary, ortalı (`if st.button("Analiz Et", type="primary"):`)
4. **Sonuç görüntüsü** — `st.image` tam genişlik, annotated görüntü (`st.session_state["defect_result_img"]`)
5. **Metrik satırı** — 4 sütun `st.metric`:
   - Tespit Sayısı
   - Anomali Skoru (0–100)
   - Severity (Düşük / Orta / Yüksek)
   - Öneri (rutin / yakın bakım / acil)
6. **Tespit tablosu** — `st.dataframe(create_metrics_dataframe(detections))` — sütunlar: Sınıf, Güven, Bbox
7. **Boş durum** — dosya yüklenmemişse: info box + desteklenen formatlar + tespit edilen sınıf listesi

**Sidebar:** `render_sidebar(active_page="defect")` + Ayarlar bölümü (güven slider, maske/bbox checkbox) + Model Durumu kartı

**Model yükleme:** `@st.cache_resource` — `yolo11n-seg.pt` fallback, başarısız olursa demo modu (görseli olduğu gibi döndürür, uyarı mesajı gösterir)

---

### 4.3 Cevher Ön Seçimi (`pages/ore.py`)

**Önemli:** Aynı button-triggered akış. Mevcut pie chart kaldırılır — sadece bar chart kalır (sınıf dağılımı). Mevcut iki sütunlu layout yerine tam genişlik (üst/alt) layout kullanılır.

**Import:**
```python
from enterprise_vision_ai.utils.image_utils import load_image, preprocess_for_model
from enterprise_vision_ai.utils.visualization import draw_annotations
from enterprise_vision_ai.utils.metrics import create_metrics_dataframe
```

**Layout (üst → alt):**

1. **Başlık bloku** — `st.title("💎 Cevher Ön Seçimi")` + açıklama satırı
2. **Yükleme alanı** — `st.file_uploader` (JPG, JPEG, PNG)
3. **"Analiz Et" butonu** — primary, ortalı
4. **Sonuç görüntüsü** — `st.image` tam genişlik, annotated görüntü
5. **Metrik satırı** — 4 sütun `st.metric`:
   - Toplam Tespit
   - Metal Oranı (%)
   - Diverter Önerisi (İleri / Kontrol / Atık)
   - Dominant Sınıf
6. **Bar chart** — `st.bar_chart` — sınıf dağılımı (manyetit, krom, atık, düşük tenör). Pie chart yok.
7. **Boş durum** — info box + cevher sınıfları açıklaması + metal oranı formülü

**Sidebar:** `render_sidebar(active_page="ore")` + Ayarlar (güven slider, maske/bbox) + Model Durumu kartı + Sınıflar kartı

---

## 5. Gradio HF Space

**Dosya:** `huggingface_space/app.py` (sıfırdan — mevcut `gr.TabbedInterface` nesting tamamen silinir)

### 5.1 Yapı

```python
with gr.Blocks(theme=gr.themes.Base(), css=CUSTOM_CSS) as demo:
    # Header HTML
    gr.HTML(header_html)

    # Ana Tabs
    with gr.Tabs():
        with gr.Tab("🔍 Defekt Tespiti"):
            gr.Markdown("...")
            with gr.Row():
                with gr.Column():
                    defect_img_input = gr.Image(type="pil", label="Görüntü Yükle")
                    defect_conf_slider = gr.Slider(0.1, 1.0, value=0.25, label="Güven Eşiği")
                    defect_btn = gr.Button("Analiz Et", variant="primary")
                with gr.Column():
                    defect_img_output = gr.Image(type="pil", label="Sonuç")
                    defect_results_md = gr.Markdown()
            defect_btn.click(
                fn=run_defect_detection,
                inputs=[defect_img_input, defect_conf_slider],
                outputs=[defect_img_output, defect_results_md]
            )
            gr.Examples(
                examples=[["examples/defect_sample.jpg"]],
                inputs=defect_img_input
            )

        with gr.Tab("💎 Cevher Ön Seçimi"):
            gr.Markdown("...")
            with gr.Row():
                with gr.Column():
                    ore_img_input = gr.Image(type="pil", label="Görüntü Yükle")
                    ore_conf_slider = gr.Slider(0.1, 1.0, value=0.25, label="Güven Eşiği")
                    ore_btn = gr.Button("Analiz Et", variant="primary")
                with gr.Column():
                    ore_img_output = gr.Image(type="pil", label="Sonuç")
                    ore_results_md = gr.Markdown()
            ore_btn.click(
                fn=run_ore_classification,
                inputs=[ore_img_input, ore_conf_slider],
                outputs=[ore_img_output, ore_results_md]
            )
            gr.Examples(
                examples=[["examples/ore_sample.jpg"]],
                inputs=ore_img_input
            )

    with gr.Accordion("📡 API Kullanımı", open=False):
        gr.Markdown(api_docs_md)

demo.launch()
```

**Örnek görseller:** `huggingface_space/examples/` klasörü oluşturulur. `defect_sample.jpg` ve `ore_sample.jpg` — küçük (200x200 px) placeholder görüntüler oluşturulur (numpy ile düz renk veya gradient).

### 5.2 API Özelliği

`gr.Blocks` ile her event handler kendi API route'unu expose eder. `/run/predict` değil, component isimlerine göre otomatik oluşturulur. `api_docs_md` içinde:
- `demo.queue()` çağrısı gereklidir (API erişimi için)
- Python `gradio_client` örneği (GradioClient kullanımı)
- curl örneği `/run/predict` endpoint'i ile
- HF Space API URL formatı: `https://<user>-<space>.hf.space/run/predict`
- `demo.queue()` sonra `demo.launch()` sırası

### 5.3 Model Yoksa Demo Modu

Her iki fonksiyon (`run_defect_detection`, `run_ore_classification`) model yüklenemezse: orijinal görüntüyü döndürür + Markdown'da "⚠️ Model yüklenemedi — demo modu" mesajı. Sahte sonuç üretilmez.

---

## 6. Dokunulmayan Dosyalar

| Dosya | Neden |
|-------|-------|
| `pages/_sidebar.py` | CSS + navigasyon güncel, değişiklik gerekmez |
| `pages/01_Defekt_Tespiti.py` | Sadece `render()` çağırıyor |
| `pages/02_Cevher_On_Secimi.py` | Sadece `render()` çağırıyor |
| `huggingface_space/requirements.txt` | Değişmez |
| `huggingface_space/hardware.yaml` | Değişmez |
| `huggingface_space/README.md` | Değişmez |
| `services/utils.py` | Deprecated shim — import edilmez, silinmez |

---

## 7. Başarı Kriterleri

- [ ] `uv run streamlit run app.py` — ana sayfa açılır, iki modül kartı görünür, sidebar `pages._sidebar` üzerinden render edilir
- [ ] Defekt sayfasına geçilir, görüntü yüklenir, "Analiz Et" butonuna basılır, sonuç (annotated görüntü + metrikler + tablo) butonun altında gösterilir
- [ ] Cevher sayfasına geçilir, aynı button-triggered akış çalışır, bar chart görünür (pie chart yok)
- [ ] Sidebar tüm sayfalarda tutarlı, eski Streamlit navigasyon linkleri görünmez
- [ ] `python huggingface_space/app.py` lokal çalışır, iki tab görünür, API accordion açılır
- [ ] Gradio'daki Python örneği `gradio_client` veya curl ile `/run/predict` URL'si gösterir
- [ ] `uv run pytest tests/ -q` — mevcut 29 test geçer, yeni test eklenmez
