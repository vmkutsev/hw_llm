| Запуск | Статус | Лимит, с | Факт, с | Токены/с | S к baseline | Peak allocated, GiB | Peak reserved, GiB | Validation loss | Шаги |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | completed | 300 | 300.20 | 6860.0 | 1.000 | 41.24 | 43.44 | 6.64258 | 360 |
| `01_batch64_acc1` | failed | 90 | — | — | — | — | — | — | — |
| `02_flash` | completed | 90 | 90.65 | 7417.9 | 1.081 | 39.29 | 41.01 | 8.10980 | 116 |
| `03_checkpoint` | completed | 90 | 90.83 | 5770.7 | 0.841 | 17.39 | 23.97 | 8.24166 | 90 |
| `04_offload` | failed | 90 | — | — | — | — | — | — | — |
| `05_flash_padding_free` | completed | 90 | 90.19 | 18441.1 | 2.688 | 15.11 | 21.24 | 6.95708 | 291 |
| `06_flash_packing` | completed | 90 | 90.54 | 21745.5 | 3.170 | 40.15 | 76.20 | 8.22593 | 61 |
| `07_flash_packing_liger` | completed | 90 | 90.02 | 25541.4 | 3.723 | 25.82 | 33.15 | 8.19009 | 67 |
| `08_flash_packing_liger_compile` | failed | 90 | — | — | — | — | — | — | — |
| `09_best_batch64_acc1` | completed | 90 | 90.11 | 26333.5 | 3.839 | 41.94 | 58.50 | 8.17864 | 69 |
| `09_best_batch128_acc1` | failed | 90 | — | — | — | — | — | — | — |
| `final` | completed | 300 | 301.14 | 26668.7 | 3.888 | 41.95 | 78.25 | 6.15063 | 246 |
