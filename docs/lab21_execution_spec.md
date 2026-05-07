# Lab 21 - Spec thực thi

## 1. Mục tiêu đầu ra

Mục tiêu của bài lab này là đạt **100 điểm chính + 15 điểm bonus = 115 điểm** với cấu hình:

- Hình thức nộp: **GitHub + HuggingFace Hub**.
- Dataset: **Option A - dataset mẫu Vietnamese Alpaca**.
- Model chính: **Llama 3.2 3B Instruct**, key HF/Unsloth dự kiến: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`.
- GPU: **Free Colab T4 16 GB**.
- Thực nghiệm rank: `r=8`, `r=16`, `r=32`, `r=64`.
- Bonus stretch goals: **target all layers**, **W&B tracking**, **HF Hub upload + model card**.
- Đánh giá qualitative: **5 prompts** so sánh side-by-side, đúng theo rubric.

Definition of done:

- Notebook chạy end-to-end trên T4.
- Train đủ các rank bắt buộc `r=8`, `r=16`, `r=64`.
- Train thêm `r=32` để có quan sát về diminishing returns.
- Train thêm 1 adapter all-layers để lấy bonus stretch goal.
- Có `rank_experiment_summary.csv`, `qualitative_comparison.csv`, `loss_curve.png`, W&B run link, HF Hub adapter link, GitHub repo link.
- `REPORT.md` đủ 6 sections theo rubric và có conclusion về rank trade-off tối thiểu 100 từ.

## 2. Ma trận thực nghiệm

### So sánh rank bắt buộc và mở rộng

Tất cả các run trong bảng này phải giữ nguyên dataset, split, model, learning rate, epochs, batch size, scheduler, seed. Chỉ thay `rank` và `lora_alpha`.

| Run ID | Mục đích | Model | Target Modules | Rank | Alpha | Bắt buộc? |
|---|---|---|---|---:|---:|---|
| base | Base perplexity + qualitative baseline | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | none | - | - | Có |
| qv-r8 | So sánh low-rank | same | `["q_proj", "v_proj"]` | 8 | 16 | Có |
| qv-r16 | Baseline theo spec lab | same | `["q_proj", "v_proj"]` | 16 | 32 | Có |
| qv-r32 | Điểm đo thêm cho diminishing returns | same | `["q_proj", "v_proj"]` | 32 | 64 | Thêm |
| qv-r64 | So sánh high-rank | same | `["q_proj", "v_proj"]` | 64 | 128 | Có |

### Thực nghiệm bonus all-layers

Run all-layers dùng để so sánh với `qv-r16`, vì cùng rank nên dễ giải thích tác động của target modules.

| Run ID | Mục đích | Model | Target Modules | Rank | Alpha | Bắt buộc? |
|---|---|---|---|---:|---:|---|
| all-r16 | Stretch goal: target all layers | same | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` | 16 | 32 | Bonus |

Khuyến nghị rank chính sẽ dựa trên `qv-r8`, `qv-r16`, `qv-r32`, `qv-r64`. Phân tích bonus sẽ so sánh `qv-r16` với `all-r16`.

## 3. Cấu hình training cố định

Dùng cấu hình này cho mọi LoRA run, trừ khi cần giảm tải để tránh OOM trên T4:

| Setting | Value |
|---|---|
| Dataset | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` |
| Sample size | Khuyến nghị 300 examples, chấp nhận 100-500 |
| Split | 90% train, 10% eval, seed `42` |
| Format | Alpaca `instruction`, `input`, `output`, render thành một field `text` |
| Max sequence length | p95 làm tròn lên power of 2, cap `1024` |
| Epochs | `3` |
| Learning rate | `2e-4` |
| LR scheduler | `cosine` |
| Warmup ratio | `0.10` |
| Train batch | `per_device_train_batch_size=1` |
| Gradient accumulation | `8` |
| Effective batch size | `8` |
| Optimizer | `adamw_8bit` |
| Weight decay | `0.01` |
| Packing | `False` |
| Eval during training | `no` |
| Eval batch | `1`, có safe fallback manual eval |
| Gradient checkpointing | enabled |
| Seed | `42` |
| Tracking | Bật W&B cho chuỗi run cuối |

Các nút fallback trên T4, theo thứ tự:

1. Giảm `MAX_SEQ_CAP` từ `1024` xuống `768` hoặc `512`.
2. Giữ `packing=False`.
3. Cleanup giữa mỗi run: `del trainer, model`; `gc.collect()`; `torch.cuda.empty_cache()`.
4. Nếu `all-r16` OOM, giảm max sequence length cho all-layers run và ghi rõ trong report.
5. Không giảm epoch/rank của 3 run bắt buộc trừ khi bất khả kháng; nếu có, ghi rõ limitation.

## 4. Metrics cần ghi lại

Mỗi adapter run phải ghi các cột sau vào `results/rank_experiment_summary.csv`:

| Column | Ý nghĩa |
|---|---|
| `run_id` | `qv-r8`, `qv-r16`, `qv-r32`, `qv-r64`, `all-r16` |
| `model_name` | Exact model key used |
| `target_modules` | `qv` hoặc `all` |
| `rank` | LoRA rank |
| `alpha` | LoRA alpha |
| `trainable_params` | Số trainable parameters |
| `trainable_percent` | Tỷ lệ phần trăm trainable params |
| `train_time_min` | Thời gian training wall-clock |
| `peak_vram_gb` | `torch.cuda.max_memory_allocated() / 1e9` |
| `eval_loss` | Eval loss từ `safe_evaluate()` |
| `eval_perplexity` | `exp(eval_loss)` |
| `final_train_loss` | Train loss cuối cùng được log |
| `hf_adapter_url` | HF Hub URL nếu đã upload |
| `wandb_run_url` | W&B run URL |
| `notes` | OOM fallback, anomaly, hoặc ghi chú diễn giải |

Base model row:

- Include `run_id=base`.
- Để trống các field chỉ áp dụng cho training.
- Compute `eval_loss` và `eval_perplexity` trên cùng eval set.
- Dùng row này để hỗ trợ so sánh quantitative "before vs after".

## 5. Spec đánh giá qualitative

Rubric chỉ yêu cầu tối thiểu 5 prompts, nên giữ 5 prompts nhưng chọn có chủ đích để bao phủ nhiều hành vi.

| Prompt ID | Category | Goal |
|---|---|---|
| P1 | Vietnamese explanation | Kiểm tra khả năng giải thích dễ hiểu |
| P2 | Code generation | Kiểm tra structured reasoning + format |
| P3 | List/format following | Kiểm tra instruction-following |
| P4 | Domain-style Vietnamese answer | Kiểm tra style tiếng Việt sau fine-tune |
| P5 | Edge case / refusal / uncertainty | Kiểm tra việc không trả lời quá tự tin khi thiếu thông tin |

Output file `results/qualitative_comparison.csv` cần có các cột:

| Column | Ý nghĩa |
|---|---|
| `prompt_id` | P1-P5 |
| `prompt` | Test prompt |
| `base_output` | Câu trả lời của base model |
| `qv_r16_output` | Câu trả lời của baseline adapter |
| `best_rank_output` | Câu trả lời của best rank adapter, nếu best rank không phải r16 |
| `winner` | `base`, `qv-r16`, `best-rank`, hoặc `tie` |
| `score_format` | 1-5 |
| `score_helpfulness` | 1-5 |
| `score_vietnamese_quality` | 1-5 |
| `comment` | Nhận xét qualitative ngắn |

Report nên trình bày 5 examples side-by-side. Không cherry-pick toàn case win; nên có ít nhất 1 case "same" hoặc "degraded" nếu có.

## 6. Kế hoạch W&B tracking

W&B là bonus stretch goal và giúp report trông chuyên nghiệp hơn.

Setup:

- Set `report_to="wandb"` sau khi W&B login sẵn sàng.
- Dùng project name: `lab21-lora-rank-experiment`.
- Dùng run names khớp với `run_id`: `qv-r8`, `qv-r16`, `qv-r32`, `qv-r64`, `all-r16`.
- Log config: model, dataset, sample size, max sequence length, rank, alpha, target modules, seed.

Artifacts cần capture:

- Training loss curve cho từng run.
- Final eval loss và perplexity.
- Peak VRAM và train time có thể log thủ công.
- W&B project/run links cần được đưa vào `LINKS.md` và `REPORT.md`.

Nếu W&B login fail trên Colab, tiếp tục training và ghi rõ rằng W&B không bật được. Bonus HF Hub vẫn quan trọng hơn.

## 7. Kế hoạch HuggingFace Hub và GitHub

### HF Hub

Push publicly ít nhất adapter tốt nhất. Để bài nộp Option B mạnh hơn, push:

- `lab21-llama32-3b-qv-r16`
- `lab21-llama32-3b-best-rank`
- `lab21-llama32-3b-all-r16`

Mỗi repo nên có:

- adapter weights,
- adapter config,
- tokenizer chỉ khi workflow save adapter cần,
- model card ngắn có dataset, base model, LoRA rank, target modules, training command/notebook link, và evaluation summary.

### GitHub

Repo hoặc submission folder nên gồm:

```text
lab21_<MSSV>/
├── REPORT.md
├── notebook.ipynb
├── LINKS.md
└── results/
    ├── rank_experiment_summary.csv
    ├── qualitative_comparison.csv
    └── loss_curve.png
```

`LINKS.md` nên chứa:

- GitHub repository URL.
- HF Hub adapter URLs.
- W&B project/run URLs.
- Colab notebook URL nếu share.

## 8. Thứ tự thực thi trên Free Colab T4

Thứ tự khuyến nghị:

1. Mở notebook trong Colab và bật T4 GPU.
2. Cài dependencies và verify CUDA.
3. Đổi `MODEL_NAME` thành `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`.
4. Load dataset sample, clean, format, tokenize, compute p95, split 90/10.
5. Evaluate base model perplexity trên eval set nếu memory cho phép.
6. Train `qv-r16` trước vì đây là rubric baseline.
7. Save adapter `qv-r16` ngay lập tức và chạy safe eval.
8. Train `qv-r8`, save, safe eval.
9. Train `qv-r32`, save, safe eval.
10. Train `qv-r64`, save, safe eval.
11. Train `all-r16`, save, safe eval.
12. Generate qualitative outputs cho base, `qv-r16`, và best rank.
13. Export CSVs và loss curve.
14. Push best adapter và bonus adapters lên HF Hub.
15. Điền `REPORT.md` và `LINKS.md`.
16. Strip notebook outputs trước khi nộp nếu file quá lớn.

Vì sao train `qv-r16` trước:

- Đây là baseline của lab.
- Nếu Colab disconnect sau đó, bạn đã có adapter cốt lõi và vẫn có thể hoàn thành một bài nộp có thể bảo vệ được.

## 9. Cấu trúc report

`REPORT.md` phải có đúng 6 section bắt buộc sau.

### 1. Setup

Cần có:

- Họ tên và MSSV.
- Submission option: Option B.
- Base model: exact model key.
- Dataset name và sample size.
- Split size.
- p95 và `max_seq_length` đã chọn.
- GPU: Tesla T4 16 GB.
- Tổng thời gian training và estimated cost.
- HF Hub, GitHub, W&B links.

### 2. Rank Experiment Results

Cần có bảng đầy đủ:

- Base.
- `qv-r8`.
- `qv-r16`.
- `qv-r32`.
- `qv-r64`.
- `all-r16` dưới dạng bonus comparison.

Các điểm cần phân tích:

- Rank nào cho perplexity thấp nhất?
- Rank nào có ROI tốt nhất?
- `r=32` có cho thấy diminishing returns trước `r=64` không?
- all-layers có cải thiện so với q/v-only ở cùng rank không?

### 3. Loss Curve Analysis

Cần có:

- `loss_curve.png`.
- Train loss có giảm đều không?
- Eval loss có gợi ý overfitting không?
- Có nhiễu nào do eval set nhỏ không?

### 4. Qualitative Comparison

Cần có 5 examples:

- Prompt.
- Base output.
- Fine-tuned output.
- Nhận xét ngắn.
- Winner hoặc score.

### 5. Conclusion về Rank Trade-off

Tối thiểu 100 từ. Cần trả lời:

- Rank tốt nhất cho dataset này và lý do.
- Rank cao hơn có cải thiện chất lượng đủ để xứng đáng với time/VRAM không?
- Khi nào nên chọn `r=8`, `r=16`, `r=32`, hoặc `r=64` trong production?
- all-layers có đáng dùng hơn q/v-only không?

Suggested conclusion stance nếu kết quả đi theo xu hướng thường gặp:

- `r=8` rẻ nhất và có thể đủ cho style adaptation.
- `r=16` có khả năng là default tốt nhất trên T4 vì cân bằng chất lượng/chi phí tốt.
- `r=32` có thể cải thiện perplexity nhẹ và giúp nhận diện diminishing returns.
- `r=64` có thể tốn hơn nhưng không cải thiện tương xứng trên dataset 300 samples.
- all-layers có thể tăng adaptation capacity, nhưng trên T4 có thể kém hấp dẫn nếu VRAM/time tăng quá nhiều.

### 6. What I Learned

Dùng 2-3 bullet cá nhân, ví dụ:

- Rank kiểm soát capacity trainable và trade-off VRAM/time như thế nào.
- Vì sao evaluation cần cả perplexity và qualitative samples.
- Vì sao QLoRA giúp fine-tune LLM thực tế trên T4.

## 10. Checklist chấm điểm

### 100 điểm chính

- [ ] Dataset Alpaca format đã được chuẩn bị và document.
- [ ] Token p95 analysis đã hoàn thành.
- [ ] `max_seq_length` đã được chọn và giải thích.
- [ ] Dùng cùng train/eval split trên mọi run.
- [ ] `qv-r8` đã train, save, evaluate.
- [ ] `qv-r16` đã train, save, evaluate.
- [ ] `qv-r64` đã train, save, evaluate.
- [ ] Base model eval perplexity đã compute hoặc limitation đã được document.
- [ ] Bảng rank có time, VRAM, params, eval loss, perplexity.
- [ ] Qualitative comparison có ít nhất 5 prompts.
- [ ] Report có đủ 6 sections bắt buộc.
- [ ] Conclusion có ít nhất 100 từ và giải thích rank trade-off.

### Bonus

- [ ] `qv-r32` đã train và phân tích diminishing returns.
- [ ] `all-r16` đã train và so sánh với `qv-r16`.
- [ ] Có W&B run links.
- [ ] Best adapter đã push lên HF Hub.
- [ ] HF model card có training/eval summary.
- [ ] GitHub repo hoặc submission folder có `LINKS.md`.

## 11. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Llama 3.2 3B yếu tiếng Việt hơn Qwen2.5 | Qualitative outputs có thể kém mượt hơn | Nêu rõ model choice và đánh giá trung thực; dataset vẫn là tiếng Việt |
| T4 OOM ở `r=64` hoặc all-layers | Thiếu required hoặc bonus run | Giảm max sequence length trước; ưu tiên các q/v-only required runs |
| Colab disconnect | Mất adapter/checkpoints | Save sau mỗi run; có thể mount Google Drive |
| W&B login issue | Bằng chứng bonus yếu hơn | Tiếp tục không block; include CSV/plots và HF Hub |
| HF Hub upload issue | Mất +5 bonus | Thử push best adapter trước; nếu fail thì include local artifacts |
| Eval set nhỏ gây variance | Perplexity nhiễu | Thảo luận limitation; dựa thêm vào qualitative examples |

## 12. Thay đổi tối thiểu cần làm trong notebook

Notebook T4 hiện tại đã khá gần. Các chỉnh sửa dự kiến trước khi chạy:

- Đổi `MODEL_NAME` từ `unsloth/Qwen2.5-3B-bnb-4bit` sang `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`.
- Thêm rank run cho `r=32`, `alpha=64`.
- Thêm argument `target_modules` tổng quát để `all-r16` có thể dùng all layers.
- Thêm W&B tracking toggle và run names.
- Thêm base model evaluation row nếu T4 memory cho phép.
- Thêm HF Hub push cells cho best adapter và bonus adapter.
- Đảm bảo `rank_experiment_summary.csv` có đủ các cột bắt buộc trong spec này.
