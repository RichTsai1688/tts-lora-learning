# 第二課：在 Colab 訓練一個 LoRA

1. 準備 Google Drive 私有資料夾，放 `manifest.csv`、相對路徑的 WAV、獨立 `reference.wav`。參考第一課，使用授權的單一說話者資料。
2. 用 [Colab](https://colab.research.google.com/) 開啟本專案的 `colab/train_lora.ipynb`；選 GPU 執行階段，依序執行並修改資料夾設定。基礎模型與 tokenizer 會在執行時從 Hugging Face 下載，請先確認其授權條款、使用條件與網路可用性。磁碟及 GPU 資源依 Colab 供應情況而定。
3. 資料檢查與人工驗收完成後才做 codec 編碼。先將 `SMOKE` 設成 `True` 執行少量批次；通過後重新啟動或調整 `RUN_NAME` 避免輸出目錄衝突，設為 `False` 訓練。fp16 是預設，支援 bf16 的 GPU 可選 bf16。實際記憶體不足時縮小資料長度或更換 GPU；不是每種免費 GPU 都能完成。
4. `loss_history.csv` 有訓練、驗證與測試損失；以驗證損失選 epoch，再以獨立測試和人工聽測比較。少量 smoke test 的結果不能代表模型品質。

程式以 LoRA 訓練，最後會**合併**權重輸出 `final_model/`，並不是獨立小型 LoRA adapter；輸出包含模型權重，可能較大。原專案的 trainer 將固定參考聲音向量寫入 speaker slot 3000；本範例只針對 0.6B Base，換模型需重新確認相容性。Colab 可能中斷，本範例不提供斷點續訓。結果存私人 Drive，不提交到公開 GitHub。

本教學沒有替模型新增情緒控制表或語言 token。資料中 `style`、`language` 只用來分析；若想以語言控制或新語言擴充，需另做模型架構、分組實驗及聽測驗證。GPU 執行、下載與訓練時間／費用依環境而變。
