# Is my major worth it?

Interactive dashboard comparing admission scores with graduate salaries across 1,787 Taiwan university programs.

<img alt="Overview of the dashboard: admission percentile versus median monthly pay for 25 fields of study" src="docs/overview.png" width="960">

Does a higher admissions bar actually pay more after graduation? The app lines up the 2025 university placement results with Ministry of Labor graduate pay, and shows where each field sits relative to what the admission threshold alone would predict.

**Demo:** no public URL yet. [Run it locally](#run-locally) and open http://localhost:8501. This is a Streamlit server app, so GitHub Pages cannot host it.

## Questions it answers

- How much of the gap in median pay does the admission threshold explain? Across these 25 fields, about **R² = 0.22**.
- Which fields pay above or below that baseline (an OLS residual on median pay)?
- For a given university and department, which field it belongs to, and that field’s P25 / P50 / P75 monthly pay.
- Which fields have a tight pay band, and which have a wide one, using `(P75 − P25) / P50`.

## Data

| Piece | Source | Grain |
| --- | --- | --- |
| Admission results, academic year 114 (2025) | [University Entrance Committee, `114_result_school_data.pdf`](https://www.uac.edu.tw/114data/114_result_school_data.pdf) | 1,787 programs |
| Graduate pay, July 2025 (ROC 114/07) | [Ministry of Labor salary navigator](https://yoursalary.taiwanjobs.gov.tw/) (full-time pay by field, labor-pension contribution file) | 25 fields |
| Program → field crosswalk | [Ministry of Education `higher_bcode4.csv`](https://stats.moe.gov.tw/files/bcode/higher_bcode4.csv) (standard subject classification, 5th revision; 2016 bachelor daytime rows) | programs |

The horizontal axis is a proxy percentile: admission score divided by the weighted full mark, then ranked across programs. The PDF does not publish the College Entrance Examination Center’s official PR.

Public salary tables are released at field level. 1,716 of the 1,787 programs match a field; 1,703 also have an admission percentile and can be looked up. Salary is a field-level proxy, not a program-level or individual earnings prediction.

The admission PDF was parsed and the salary portal was scraped in a course working folder. This repository ships the dashboard, the joined tables in `data/`, `extract_quantiles.py` (P25/P50/P75 from the salary buckets), and `analysis_v2.py` (the field-level join and residual).

## Tech

Python 3.13 · Streamlit · Plotly · pandas · SciPy · scikit-learn

## Run locally

以下 PowerShell 指令使用隔離環境，不需啟用 venv 或更改 execution policy。請在自己的 clone 根目錄執行下列相對路徑指令。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 data_pipeline.py
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

Open http://127.0.0.1:8501. Five pages: overview, department lookup, field detail, pay spread, and methodology. 用 `Ctrl+C` 停止服務。

`requirements-lock.txt` 是 Windows / Python 3.13.1 本機驗證環境的完整套件版本快照，不是所有平台的相容性保證，也未鎖 wheel hash。`requirements.txt` 保留原本的寬鬆下限，更新依賴後請重跑驗證；既有 devcontainer 未列入這次驗收。啟動只讀取隨附 CSV，不需原始 HTML、爬蟲、憑證或對外 API。

### 離線重建與資料檢查

**已知限制：來源品質 FAIL，7 筆達成率超過 100% 待複核。** `--strict-quality` 目前回傳 exit 1；通過功能測試或一致性檢查不代表資料品質過關。原始資料、分數、加權滿分與排名保留原樣，未重驗上游擷取鏈或跨年份分類。

```powershell
# 僅讀取並核對；預設會列出既有來源品質待複核事項
.\.venv\Scripts\python.exe -X utf8 data_pipeline.py

# 嚴格來源品質門檻：目前隨附資料有7筆達成率>100%，預期 exit code 1
.\.venv\Scripts\python.exe -X utf8 data_pipeline.py --strict-quality

# 重建原有學門主表；輸出資料夾必須尚不存在，不覆寫來源
.\.venv\Scripts\python.exe -X utf8 analysis_v2.py --no-plots --output-dir '.\artifacts\rebuild-01'
```

驗證器檢查唯一鍵、缺值一致性、數值範圍、薪資桶與分位數、join 完整性，再由校系表重算學門 PR 中位數及校系數。以 **學門一對一** 合併薪資，保留原本以 P50 作應變數的 OLS 與殘差遞減排序，逐欄核對 `final_dataset_v2.csv`；不把學門薪資展開成校系觀測值來重複加權。主表浮點比對只容許 `1e-8` 的序列化差異；分位數整元與人數必須符合重算結果。

預設 exit code `0` 只代表隨附表格與現有方法**一致且可重現**，不代表上游擷取、跨年份分類或推論已確認正確。缺少檔案、重複鍵、未知學門、損毀值或主表不一致會 exit `1`。`--data-dir` 可指定另一組三張 CSV；`--output` 可另存重建 CSV，但拒絕覆寫任何已存在檔案。輸出父資料夾須事先存在。

這次隨附資料的核對結果：

| 項目 | 數量／處理 |
| --- | --- |
| 校系總數 | 1,787；`系組代碼` 以文字讀取，保留前導零 |
| 未對應學門 | 71；保留缺值，不補分類 |
| 缺錄取分數／達成率／PR | 15；三欄缺值一致，不補零 |
| 同時未對應且缺分數 | 2；排除聯集為 84，不能直接把 71 與 15 相加 |
| 可分析校系／學門 | 1,703／25；`dept_count` 是有有效 PR 的校系數 |
| 薪資表 | 合計列 + 27 學門；排除合計，另列出沒有錄取資料的「安全服務學門」「其他學門」 |
| 13 個薪資桶 | 各列加總等於 `contributors_n`，不是 `employed_n`；分位數是這些桶的內插估計 |
| 達成率超過 100% | 7；系組代碼 `0146, 1335, 2045, 2120, 2542, 2651, 2665`，需回原始來源複核，未改分數、加權滿分或排名 |

`風險指標` 沿用原始算法：先用未取整的 P25/P50/P75 算比值、四捨五入三位，最後才把顯示用分位數取整。開放薪資區間的 18,000／150,000 邊界也是原有假設，未以本次驗證確認其真實性。

`analysis_v2.py` 已改由隨附校系表聚合，不再依賴未收錄的 `uac_xuemen_agg.csv`。`--no-plots` 是本次實測入口；若要執行原有靜態出圖，仍需自行在 venv 安裝 `matplotlib`、`seaborn`，本次未驗出圖與字型。`extract_quantiles.py` 的 HTML 擷取另需 `beautifulsoup4` 和未收錄的 `data/scrape_frame_2.html`，不能把現有 CSV 重建當成上游爬蟲／PDF 解析已重現。

Course project at NTNU. Author: 王語揚. Advisor: 張維元.

## 中文說明

大學科系「CP 值」大檢驗：114 學年大考分發 1,787 個校系的錄取分數，對上勞動部 114 年 7 月 25 個學門的畢業生薪資，看分數能解釋多少薪資差異，以及哪些學門高於或低於模型預期。

作者：41111017E 王語揚 ・ 指導：張維元教授 ・ 課程：資料科學在教育上的應用

資料來源見上方表格（大考分發委員會錄取 PDF、勞動部薪資行情、教育部 `higher_bcode4.csv`）。PDF 解析與薪資頁面擷取在課程作業目錄完成；這個 repo 收錄互動網站與整理後的資料表。本機執行方式與英文區相同。
