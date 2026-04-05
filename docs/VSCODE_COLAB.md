# Connecting VS Code to Google Colab GPU

The integration you found allows you to edit the "Subliminal Learning" code in your local VS Code environment while running the heavy computations on Google's cloud GPUs.

## 1. Setup in Google Colab (Browser)
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** > **Change runtime type** > **GPU** (A100/L4 recommended).
3. Click the **Connect** button at the top right.
4. Once connected, click the down arrow on the Connect button and select **Connect to a local runtime**... wait, no! 
5. Select **Connect to an external runtime** or look for the **"VS Code"** option in the Colab sidebar (if you have Colab Pro) or use the **"Connect to VS Code"** command if it's visible in your region.

## 2. Setup in VS Code (Local)
1. Install the **"Google Colab"** extension in VS Code.
2. Open the Command Palette (`Cmd+Shift+P`) and type: `Colab: Connect to Google Colab`.
3. Follow the authentication flow to link your Google account.
4. **Important**: Select your active Colab runtime as the "Remote Interpreter."

## 3. Running Subliminal Learning
Now, when you open a terminal in VS Code, it will actually be running on the Colab VM! You can launch the high-res run directly:

```bash
# Running 200 samples on cloud GPU from local VS Code
# This automatically scales to 128-dim architecture on A100/H100
python run_real_mgnify.py --arch hierarchical --n-per-env 100 --epochs 20 --kl-weight 0.1 --device cuda
```

## 💾 Persistent Storage (Google Drive)
Colab runtimes are temporary. To ensure your massive sample downloads are not lost when you disconnect, you should mount your Google Drive:

1. In a Colab cell (browser), run:
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. In your VS Code terminal, create a symbolic link to your Drive:
```bash
ln -s /content/drive/MyDrive/subliminal_data ./data/samples_real
```
Now, all downloads will go directly to your Google Drive and be available for future sessions.

## Benefits
- **Persistent Files**: Your `outputs/` and `data/` will be safe on Google Drive.
- **Interactive Debugging**: Set breakpoints in `hierarchical_sample_lm.py` and they will trigger even though the code is running in the cloud.
- **No Git Friction**: Quick changes to hyperparameters don't require an intermediate GitHub push.
