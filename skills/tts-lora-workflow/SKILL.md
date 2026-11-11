---
name: tts-lora-workflow
description: Prepare and analyze authorized single-speaker speech datasets and guide a Qwen3-TTS LoRA experiment in Colab. Use for voice, speaking style, or language adaptation; do not claim unsupported languages or controlled emotions are proven.
---

# TTS LoRA 教學流程

使用者應提供本專案路徑及已授權資料。先閱讀專案 `README.md`、`docs/01-dataset.md`；要求 Colab 訓練時再讀 `docs/02-colab.md` 和 `colab/train_lora.ipynb`。只處理授權錄音，移除逐字稿中的個資；不要上傳原始音訊、逐字稿、模型或憑證至公開儲存庫。

1. 釐清目標是音色／語調或目標語言，確認說話者、資料用途與授權。記錄語言、語調及獨立錄音場次；不自行抓取或假設來源可用。
2. 依教材製作 CSV 和 24 kHz 單聲道 PCM WAV；先運行 `scripts/dataset_tool.py` 盤點，再由人聽核文字、音訊、內容與授權。依場次切分，保留獨立固定參考音檔，避免重複音訊跨集合。報告的語言與語調只供分析，不會自動控制模型。
3. 在 Colab 依筆記本逐格產生 codec、先做小型測試，再訓練一個 LoRA。遵循筆記本的資料與輸出位置、GPU 檢查；沒有實際執行就不要聲稱訓練成功。
4. 比較驗證損失和獨立測試句的人工試聽，記錄限制；目標語言的效果與是否遺忘原語言都需要實測。輸出是合併模型，非獨立 adapter；完整模型和私有資料不應上傳公開儲存庫。
