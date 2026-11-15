# TỔNG HỢP DỰ ÁN NLP-28 — nhật ký handoff

> **File này để làm gì:** đọc một lượt là hiểu dự án đang tới đâu, đã sửa gì, còn treo gì.
> **Quy ước:** mỗi lần sửa code / dữ liệu / cấu hình thì **cập nhật thêm file này**.
> **Cập nhật lần cuối:** 2026-11-15 · commit `452de77` · 40/40 test pass · 48 tệp trong git

---

## 1. Tóm tắt một đoạn

Sửa lỗi & chuẩn hóa phụ đề tiếng Việt từ ASR bằng Seq2Seq (mô hình nền `bmd1905/vietnamese-correction-v2`,
GPU Colab). Đã train 4 lần (133 → 10k → 100k câu). **Sau Lần 4 phát hiện 7 vấn đề ở khâu
dữ liệu và khâu đo** — không phải ở kiến trúc mô hình; cả 7 đều đã sửa, và sửa đó là phần
lớn giá trị của giai đoạn này.

Phát hiện lớn nhất: **Whisper thực ra GIỮ dấu thanh (99,3%)** — nó sai dấu và nhầm âm, chứ
không bỏ dấu. Cả 4 lần train trước đều luyện cho tình huống mất dấu, tức **sai bài toán**.

---

## 2. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Code | ✅ 40/40 test pass, đã push (`452de77`) |
| Notebook Lần 5 | ✅ 13 cell, hợp lệ với Python 3.11 (Colab), comment đã rút gọn |
| Tập test | ✅ 5 tập trong git, 2.981 câu |
| Train Lần 5 | ⏸ **chưa chạy** — đang chờ bật Colab |
| Báo cáo | Đề cương Mục 10 (tổng quan) + `BAO_CAO_NLP28` (chi tiết, không push) |
| Đang treo | `corpus_noisy` chưa đưa vào chỗ đo; có tách notebook đôi không |

---

## 3. Bản đồ repo

```
src/data_processing/
  noise_generator.py    4 hồ sơ nhiễu + ASRNoiseGenerator (có seed tái lập)
  corpus_builder.py     khử trùng lặp, chia tập theo hash, sinh cặp, lấy Wikipedia
  build_basic_test.py   dựng tập test gd1 (ShynBui + Wikipedia)
  build_corpus_test.py  dựng tập control từ corpus/ (văn học)
  build_kichban_test.py dựng tập test gd2 từ phụ đề ASR thật
  dataset_loader.py     pad ĐỘNG theo batch (thay vì cứng 128)
src/models/
  train_seq2seq.py      chống đè checkpoint, manifest, --resume, --label_smoothing
  inference.py          clean_hallucinated_tail + set_decoding() không cần nạp lại
src/evaluation/
  metrics.py            WER, BLEU, CER, Exact Match, ROUGE-L, Edit-P/R/F1
  run_benchmark.py      bảng tổng hợp + bảng theo LOẠI LỖI + bảng theo NGUỒN + log .txt/.json
  compare_runs.py       đặt nhiều đợt train cạnh nhau trên cùng thước
data/gold_test/         xem Mục 4 (có README riêng giải thích từng tập)
notebooks/              train_3_10k, train_4_100k (hồ sơ), train_5_optimize (dùng tiếp)
```

---

## 4. Dữ liệu

### Train / Val (sinh ra khi chạy, KHÔNG nằm trong git)

| | |
|---|---|
| Nguồn | ShynBui, 220k câu nạp dư → khử trùng lặp → cắt còn đúng **100.000 câu sạch** |
| Chia tập | theo **hash nội dung** (không theo vị trí dòng) → train 90.028 / val 9.972, **trùng 0** |
| Hồ sơ nhiễu | `legacy`, `asr_hard`, `whisper_like` |
| Nơi lưu | Colab `/content/.../data/synthetic/` (tạm) + Drive `vi-sub-correction/data/` (bền) |

### Test (nằm trong git, đã đóng băng)

| Nhóm | Tệp | Câu | Nguồn | Dùng để |
|---|---|---:|---|---|
| gd1 | `sample_test.jsonl` | 25 | ShynBui | đối chiếu số cũ |
| gd1 | `basic_test_unaccent_all.jsonl` | 200 | ShynBui | **bài toán gốc** |
| gd1 | `basic_test_asr_hard.jsonl` | 200 | ShynBui | nhiễu hỗn hợp |
| gd1 | `basic_test_multisource.jsonl` | 300 | ShynBui + **Wikipedia** | phát hiện lệch nguồn |
| gd2 | `kichban_test.jsonl` | 132 | **ASR thật** | nâng cao |
| control | `corpus_identity.jsonl` | 962 | văn học | đo over-correction |
| control | `corpus_noisy.jsonl` | 962 | văn học | ngoài miền |

`corpus/` (văn bản gốc) và SRT thô **không nằm trong git** vì bản quyền — để trên Drive.

---

## 5. Bảy phát hiện sau Lần 4

| Mã | Vấn đề | Con số |
|---|---|---|
| P1 | Val rò rỉ vào train do chia theo **vị trí dòng** trên dataset trùng lặp nội bộ | **20,8%** (1.894/9.121) |
| P2 | Phân bố nhiễu train không khớp ASR thật | train 0% câu mất dấu · Whisper thật 99,3% **giữ** dấu |
| P3 | Tập test quá nhỏ, một loại lỗi, không có câu sạch | 25 câu = 4%/câu |
| P4 | 79% token là padding thừa, `DataCollatorForSeq2Seq` bị vô hiệu hoá | pad cứng 128, thực tế ~27 |
| P5 | Không có cặp `clean→clean` nào trong train | 0% |
| P6 | Tham số giải mã chồng hai hình phạt lặp | `rep_pen 1,2` **và** `no_repeat_ngram 3` |
| P7 | `benchmark_summary.md` trong git chứa số của **mô hình gốc** | BLEU 0,00 / WER 92,00% |

Hệ quả của P1: `eval_loss 0,01307` (thấp hơn cả `train_loss 0,03736`) — một phần vì val
chứa bản sao của train. Mục 8 đề cương dùng số này làm bằng chứng "hội tụ", nay phải đọc
lại kèm cảnh báo.

---

## 6. Đã sửa gì

| Tệp | Sửa |
|---|---|
| `corpus_builder.py` | **Mới.** Khử trùng lặp + chia theo hash + **tự chặn nếu còn rò rỉ**; `max_clean`; lấy Wikipedia; trường `source` |
| `noise_generator.py` | **seed tái lập** (trước đây mỗi lần chạy ra tập train khác!); sắp xếp nhãn lỗi (trước phụ thuộc `PYTHONHASHSEED`); hồ sơ `whisper_like` |
| `dataset_loader.py` | Pad động theo batch (P4) |
| `metrics.py` | Thêm CER, Exact Match, ROUGE-L, đếm theo câu |
| `run_benchmark.py` | Bảng theo loại lỗi + theo nguồn; xuất `.txt` và `.json` |
| `train_seq2seq.py` | Chống đè (`_v2`), `manifest.json`, `--resume`, `--label_smoothing`, `--num_train_split`, lưu log train |
| `inference.py` | `set_decoding()` đổi tham số giải mã không phải nạp lại 1,6 GB |
| `compare_runs.py` | **Mới.** Bảng A/B nhiều đợt |

**Hai lỗi tái lập do tôi tự gây ra và tự bắt được** (đáng nhớ vì cùng loại):
1. `ASRNoiseGenerator` dùng `random` toàn cục → mỗi lần chạy sinh tập train khác nhau.
2. `list(set(errors))` → thứ tự phụ thuộc `PYTHONHASHSEED`, tệp JSONL khác nhau về byte.

---

## 7. Hai bài toán — không phải một

| | **gd1** | **gd2** |
|---|---|---|
| Đầu vào | mất dấu hoàn toàn | có dấu nhưng **sai** |
| Việc phải làm | **thêm** dấu | **phát hiện** chỗ sai rồi sửa |
| Bản chất lỗi | chính tả / dấu câu | **ngữ âm** |
| Nhiễu train | `asr_hard` | `whisper_like` |
| Tập test | `gd1_basic/` | `gd2_real_asr/` |

**Giỏi cái này không suy ra giỏi cái kia.** Nên train riêng hai đợt, không trộn — trộn là
đánh đổi lấy tiết kiệm thời gian, không lấy chất lượng.

Trong notebook Lần 5, đổi bài toán bằng **2 dòng**:

| Cell | gd1 | gd2 |
|---|---|---|
| 5 | `GRID_VAL = VAL_HARD` | `VAL_WHISPER` |
| 8 | `VARIANT = 'asr_hard'` | `'whisper_like'` |

---

## 8. Quy trình chạy

```
sửa .py trong repo → pytest → commit + push → Colab git pull → bấm Run
```

**Không sửa tay trên Colab.** Lần 4 đã trả giá: sửa `NUM_SAMPLES` trên Colab, không kéo về,
nên bản trong repo vẫn ghi 10.000 và không tái lập được.

**Hồ sơ nhiễu — số đo trên 500 câu cùng nguồn:**

| Hồ sơ | Câu giữ dấu thanh | Câu giữ nguyên | Dùng cho |
|---|---:|---:|---|
| `legacy` | 100% | 0,2% | như Lần 1–4 (đối chứng) |
| `unaccent_all` | 0% | 0% | sinh test gd1 |
| `asr_hard` | 24,2% | 11,6% | train gd1 |
| `whisper_like` | **100%** | 46,2% | train gd2 |
| *kichban (ASR thật)* | *99,3%* | *54,5%* | làm mốc so sánh |

---

## 9. Checklist trước khi train trên Colab

- [ ] **Dùng tài khoản gg2** — gg1 đã hết hạn mức GPU (CPU không train nổi: ~40–60 giờ)
- [ ] **Share Drive gg1 → gg2**: chuột phải thư mục `vi-sub-correction` → Chia sẻ → email gg2,
      quyền **Người chỉnh sửa** → trên Colab gg2: Drive → *Được chia sẻ với tôi* → **Thêm lối tắt vào Drive của tôi**
- [ ] Trên Drive phải có: `bmd1905_checkpoint.zip` (mốc Lần 4) và `corpus/` (nếu chạy cell 4)
- [ ] `git pull` để lấy code mới nhất
- [ ] (Nên) chạy thử nhanh 10% dữ liệu trước: thêm `--num_train_split 0.1` vào cell 8,
      mất vài phút thay vì 1–2 giờ, để phát hiện lỗi đường dẫn

**Thứ tự chạy:** cell 0 → 10 cho đợt gd1. Mỗi đợt train tự đo cả 6 tập test.

---

## 10. Bẫy đã trả giá (đừng lặp lại)

| Bẫy | Hậu quả thật |
|---|---|
| Chia tập theo vị trí dòng | Val rò rỉ 20,8% → eval_loss đẹp giả tạo |
| Nhiễu không seed | Mỗi lần chạy ra tập train khác → không tái lập |
| `list(set(...))` trong JSONL | Tệp khác nhau về byte giữa các tiến trình |
| `load_dotenv()` không `override=True` | Biến môi trường hệ thống đè `.env` → nạp nhầm mô hình |
| Ghi checkpoint vào `/content` | Rớt runtime là mất trắng 2 giờ (Lần 4 suýt dính) |
| Sinh test bằng chính hồ sơ nhiễu của train | Đo "thuộc lòng hàm nhiễu nhà mình", không phải năng lực |
| Test bằng index thay vì anti-join nội dung | 1.479/2.000 câu dính vào tập train |
| Kết luận từ 25 câu | 1 câu lệch = 4% chỉ số |

---

## 11. Lịch sử commit

```
452de77  feat(test): Wikipedia cho gd1, bảng theo nguồn, rút gọn notebook
fd29a90  chore: bỏ kết quả benchmark tự động ra khỏi git
98b68cf  chore: đưa PDF, script export, .code-workspace ra khỏi git
ba3714e  feat(notebook): chọn bài toán bằng một dòng, đo mọi tập test
0a221a7  feat(data,eval): chẩn đoán Lần 4, whisper_like, tổ chức lại tập test
78987ab  feat(notebook): add benchmark evaluation step   <- trước khi bắt đầu dọn dẹp
```

---

## 12. Việc còn treo

| Việc | Ghi chú |
|---|---|
| Train Lần 5 (gd1) | Chưa chạy. Notebook đã sẵn sàng |
| Train Lần 6 (gd2) | Đổi 2 dòng ở cell 5 và 8 |
| `corpus_noisy` | Chưa đưa vào `TEST_SUITES` — thêm 1 dòng là xong |
| Đọc `kichban_2026-10-25.srt` gốc | 1 phút, xác nhận P2 (đã xác nhận qua bản tải về) |
| Mục 7 đề cương | `"tổ buồn 247" → "tổ chức 247"` **không phải over-correction** mà là lỗi nhận diện tên riêng — cần sửa lại mô tả |
| Bảng so sánh 4 lần train | Lần 1/2A/2B/3 chưa từng benchmark → chỉ Lần 4 có số |

---

## Cách cập nhật file này

Khi có thay đổi, sửa đúng các mục liên quan:

```
Mục 2  Trạng thái       — đổi khi xong/treo việc gì
Mục 5  Phát hiện        — thêm mã mới (P8, P9...) khi tìm ra vấn đề
Mục 6  Đã sửa gì        — thêm dòng cho tệp vừa sửa
Mục 11 Lịch sử commit   — thêm commit mới lên đầu
Mục 12 Việc còn treo    — thêm/gạch việc
```

Và cập nhật dòng **"Cập nhật lần cuối"** ở đầu file.
