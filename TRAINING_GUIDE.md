# Training Guide: Create and Train Your Own Model

This guide explains the full process to build a custom dataset and train a model for point activity detection.

## 1) Folder Structure

Use a structure like this:

```text
bt/
  src/
    source/
      full_video_1.mp4
      full_video_2.mp4
      labeled_data/
        full_video_1.json
        full_video_2.json
      short_clips/
        short_rally/
        long_rally/
        break/
        towel_break/
```

Notes:

- Full videos are your primary training data.
- Short clips are useful for hard-case review and future feature/model improvements.

## 2) Label Format for Full Videos

Each full video label file should contain:

```json
{
  "video": "E:/personalprojs/bt/src/source/full_video_1.mp4",
  "points": [
    { "start_sec": 18.45, "end_sec": 34.07 },
    { "start_sec": 44.06, "end_sec": 55.29 }
  ]
}
```

Rules:

- `start_sec` and `end_sec` are seconds from the start of the video.
- Points must be sorted by time.
- Points must not overlap.
- Always `start_sec < end_sec`.

## 3) Build `train_items.json`

`src/build_training_npz.py` expects a top-level list of video items.

Example `src/train_items.json`:

```json
[
  {
    "video": "E:/personalprojs/bt/src/source/full_video_1.mp4",
    "points": [
      { "start_sec": 18.45, "end_sec": 34.07 },
      { "start_sec": 44.06, "end_sec": 55.29 }
    ]
  },
  {
    "video": "E:/personalprojs/bt/src/source/full_video_2.mp4",
    "points": [
      { "start_sec": 8.15, "end_sec": 15.12 },
      { "start_sec": 27.05, "end_sec": 32.0 }
    ]
  }
]
```

## 4) Create Environment and Install Dependencies

From project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 5) Build Training Dataset (`.npz`)

### Option A: From Full Videos + Labeled Points

Run:

```powershell
python src/build_training_npz.py --items-json "E:\personalprojs\bt\src\source\labeled_data\train_items.json" --output "E:\personalprojs\bt\src\source\dataset.npz"
```

You can optionally add short clips to boost accuracy:

```powershell
python src/build_training_npz.py --items-json "E:\personalprojs\bt\src\source\labeled_data\train_items.json" --output "E:\personalprojs\bt\src\source\dataset_breaks_and_rallies_1.npz" --rally-clips-dir "E:\personalprojs\bt\src\source\videos\short_rallies" --rally-clips-dir "E:\personalprojs\bt\src\source\videos\long_rallies" --break-clips-dir "E:\personalprojs\bt\src\source\videos\breaks"
```

### Option B: From Separate Rally/Break Directories Only

If you have rally and break videos in separate folders (no labeled points required):

```powershell
python src/build_training_npz.py --output "E:\personalprojs\bt\src\source\dataset_rb.npz" --rally-clips-dir "E:\personalprojs\bt\src\source\short_rallies" --rally-clips-dir "E:\personalprojs\bt\src\source\long_rallies" --break-clips-dir "E:\personalprojs\bt\src\source\breaks"
```

### Dataset Output

Expected output includes:

- dataset path
- total samples
- active ratio (proportion of rally frames)
- feature dimension (should be 15 with new temporal features)
- breakdown by source (full videos, rally clips, break clips)

## 6) Train the Model (`.joblib`)

### Basic Training

Run:

```powershell
python src/train_point_activity_model.py --dataset "E:\personalprojs\bt\src\source\dataset.npz" --output-model "E:\personalprojs\bt\src\source\point_activity.joblib"
```

### Advanced: Optimize Threshold for Rally Detection

For better rally-only output, automatically search for the best decision threshold:

```powershell
python src/train_point_activity_model.py --dataset "E:\personalprojs\bt\src\source\dataset_rb.npz" --output-model "E:\personalprojs\bt\src\source\point_activity_rb.joblib" --optimize-threshold
```

This will print the suggested threshold at the end (e.g., `Suggested threshold for rally detection: 0.68 (F1=0.92)`). Use this value in the UI.

### Tuning Hyperparameters

You can adjust random forest settings:

```powershell
python src/train_point_activity_model.py --dataset "E:\personalprojs\bt\src\source\dataset.npz" --output-model "E:\personalprojs\bt\src\source\point_activity.joblib" --n-estimators 600 --max-depth 24 --min-samples-leaf 3 --optimize-threshold
```

Parameters:

- `--n-estimators`: number of trees (default: 500, higher = more accurate but slower)
- `--max-depth`: max tree depth (default: 22, higher = more complex patterns)
- `--min-samples-leaf`: min samples per leaf (default: 4, lower = more flexible)
- `--optimize-threshold`: search for best rally F1 threshold on test split

### Training Output

Expected output includes:

- classification report (precision/recall/F1 per class)
- suggested threshold (if `--optimize-threshold` used)
- feature dimension (15 with temporal features)
- saved model path

## 6.1) Use your model for chunking (rallies only)

```powershell
python src/chunk_points.py --video "E:\videos\match.mp4" --output-dir "E:\videos\chunks" --model "E:\personalprojs\bt\src\source\point_activity.joblib" --max-rally-sec 45
```

`--max-rally-sec` helps drop unrealistically long segments so break periods are not exported as rally clips.

## 7) Model Features Explained

The model now uses **15 temporal features** per frame to distinguish rallies from breaks:

**Base motion features (3):**

- Energy: average frame intensity change
- Speed: movement magnitude (pixels/frame)
- Area: proportion of motion contours

**Temporal context (6):**

- 5-frame and 15-frame rolling averages of energy, speed, and area
- Helps distinguish sustained rally motion from brief incidental movement

**Volatility (3):**

- 9-frame rolling standard deviation of energy, speed, area
- Sustained rallies have higher volatility than static breaks

**Frame deltas (3):**

- Absolute frame-to-frame changes in energy, speed, area
- Captures motion dynamics (serve, smash, movement)

This richer feature set makes the model robust to:

- Player walking between points
- Camera motion
- Crowd noise/movement
- Lighting changes
- Breaks with some movement

## 8) Labeling Best Practices for Better Accuracy

- Use 20 to 50 full match videos.
- Keep target player in bottom half for all training videos.
- Include difficult sequences: short rallies, long rallies, breaks, towel breaks, delays.
- Keep a single consistent rule for boundaries:
  - start: serve contact / first clear rally action
  - end: shuttle dead and rally clearly stopped

## 9) Common Mistakes to Avoid

- Wrong video path in JSON.
- Using object instead of list in `train_items.json`.
- Unsorted or overlapping point ranges.
- Mixing camera styles too early (start with one style, then expand).

## 10) Quick Pre-Training Checklist

- `train_items.json` is a list.
- Every video path exists.
- Every point has `start_sec` and `end_sec`.
- No overlap, sorted by start time.
- At least 2 full videos labeled correctly before first training run.
- Rally and break clips are separated into distinct folders.

## 11) Using Your Trained Model in the UI

After training, place your model file in:

```
E:\personalprojs\bt\src\source\point_activity_rb.joblib
```

Then in the Gradio UI:

1. Start the app: `python src/ui_app.py`
2. Optionally upload a custom `.joblib` model via the file input
3. Upload a video
4. Adjust threshold slider (use the value printed during training as starting point)
5. Click "Process Video"
6. Output includes both rally (`point_XX`) and break (`break_XX`) chunks with labels in `segments.json`

If a model exists at `src/source/point_activity_2.joblib` or `src/source/point_activity_rb.joblib`, it will be auto-loaded.

## 12) Chunking Full Videos via CLI

Once trained, chunk videos with rally-only output:

```powershell
python src/chunk_points.py --video "E:\videos\match.mp4" --output-dir "E:\videos\chunks" --model "E:\personalprojs\bt\src\source\point_activity_rb.joblib" --threshold 0.68 --max-rally-sec 45
```

Key flags:

- `--model`: path to your trained `.joblib` file
- `--threshold`: use the value from `--optimize-threshold` output (default: 0.5)
- `--max-rally-sec`: drop segments longer than this (default: 45s)

Output:

- `point_XX.mp4` folders: rally chunks
- `break_XX.mp4` folders: non-rally chunks
- `segments.json`: metadata with explicit labels and timestamps
