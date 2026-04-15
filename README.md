# AtlasScope Studio

AtlasScope Studio 是你自己的本機影像地理預測專案。它可以直接讀取電腦上的圖片，使用視覺分類模型推測最可能的地點，並把 Top-K 候選結果畫在互動式地圖上。

這個版本同時保留了原本的圖尋遊戲整合能力，但核心定位已經改成一個更完整的本地工具：

- 本機圖片上傳 Web UI
- 互動式地圖結果頁
- 歷史紀錄保存
- CLI 與 Web 共用同一套推論服務
- 支援較新的 backbone 架構，並保留 legacy 權重相容模式

## 現在有哪些功能

### Web UI

- 從本機選取圖片
- 選擇推論 backbone
- 顯示原圖預覽
- 顯示 Top-K 預測卡片
- 直接把預測點畫在地圖上
- 保留最近歷史結果
- 顯示每次推論的 explainability report
- 在有 API key 時用 LLM 產生自然語言解釋

### CLI

- `--image <path>` 直接對本地圖片推論
- `--game-id <uuid>` 走圖尋整合模式
- 自訂 `--backbone`、`--checkpoint`、`--topk`

### 輸出

- `outputs/predictions/*.json`
- `outputs/maps/*.html`
- `outputs/uploads/*`
- `outputs/history.json`

## 專案結構

```text
app.py               本地 Web UI
Main.py              CLI 入口
services.py          共用服務層，統一預測 / 圖尋 / 歷史 / 輸出
project_config.py    專案品牌與預設路徑設定
history.py           預測歷史紀錄
inference.py         推論流程與 PredictionBundle
visualization.py     地圖輸出與可嵌入地圖元件
Model.py             backbone 與 checkpoint 載入
TuxunAgent.py        圖尋 API 與街景抓圖
static/              Web 樣式
templates/           Web 頁面
```

## 環境建議

建議使用：

- Python `3.10` 或 `3.11`
- Windows / macOS / Linux 都可

目前這個工作區的 Python 是 `3.14`，通常不適合直接安裝 PyTorch 現成套件，所以最好另外建立虛擬環境。

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 啟動 Web UI

```bash
python app.py
```

打開瀏覽器進入：

```text
http://127.0.0.1:5000
```

然後直接選本機圖片測試即可。

## 可解釋性與 LLM

每次推論都會輸出：

- 原始模型信心
- OCR / script hint
- heuristic rerank 是否介入
- Explainability report

如果沒有設定 LLM，UI 會顯示本地 fallback 解釋。

如果你想啟用真正的 LLM 解釋，可以先設定：

```bash
set OPENAI_API_KEY=你的_api_key
set OPENAI_MODEL=gpt-4.1-mini
```

如果你用的是相容 OpenAI 格式的其他服務，也可以加：

```bash
set OPENAI_BASE_URL=https://your-endpoint/v1
```

## 使用 CLI 預測本機圖片

```bash
python Main.py --image C:\path\to\sample.jpg --backbone legacy_mobilenet_v3
```

## 使用 CLI 跑圖尋模式

先把登入 cookie 放進 `cookie.txt`，再執行：

```bash
python Main.py --game-id <GAME_ID> --backbone legacy_mobilenet_v3
```

## Backbone 說明

目前架構支援：

- `legacy_mobilenet_v3`
- `convnext_tiny`
- `efficientnet_v2_s`
- `vit_b_16`

但這個 repo 目前只有舊版 `v0.3.0.pth` 權重，所以真正可直接跑通的預設模式是：

```text
legacy_mobilenet_v3
```

如果你有自己訓練的新權重，可以這樣指定：

```bash
python Main.py --image C:\path\to\sample.jpg --backbone convnext_tiny --checkpoint models\your_convnext_checkpoint.pth
```

## 接下來還可以加什麼

這次我先把產品化基礎打好。如果你想繼續擴充，最值得加的功能是：

1. 真正的訓練腳本與驗證流程，讓 `ConvNeXt` / `ViT` 有可用權重
2. 熱點分布圖與候選點群聚，而不是只有 Top-K 點位
3. EXIF 讀取與拍攝方向輔助
4. 批次上傳資料夾，輸出整批結果報表
5. 比對真實位置後計算距離誤差
6. 生成可分享的報告頁
