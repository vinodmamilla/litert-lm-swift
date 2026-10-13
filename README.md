# LiteRT-LM Swift Pre-Release Testing (`litert-lm-swift`)

A multiplatform (iOS & macOS) test harness specifically designed for QA and release validation of candidate **LiteRT-LM** builds, validating both **Multimodal Embedding Models** and **Large Language Models (LLMs)** across Apple Silicon hardware.

---

## 📋 Pre-Release Testing Workflow

Follow these steps for pre-release validation:

### Step 1: Clear Xcode & SPM Caches
Always run these commands before testing to prevent Xcode from using cached SPM binaries:

```bash
rm -rf ~/Library/Caches/org.swift.swiftpm
rm -rf ~/Library/org.swift.swiftpm
rm -rf ~/Library/Developer/Xcode/DerivedData/*
```

### Step 2: Clone Repositories
Clone the testing app and the target release candidate branch:

```bash
git clone https://github.com/vinodmamilla/litert-lm-swift.git
git clone -b release/v0.18.0 https://github.com/google-ai-edge/LiteRT-LM.git
```

### Step 3: Open in Xcode & Add Local Dependency
1. Open `litert-lm-swift` in Xcode:
   ```bash
   cd litert-lm-swift
   open iosandmac.xcodeproj
   ```
2. In Xcode, click **File** > **Add Package Dependencies...** > **Add Local...** and select the cloned `LiteRT-LM` folder.

### Step 4: Add Prebuilts & Update `Package.swift`
1. Inside the cloned `LiteRT-LM` directory, create a `Prebuilts` folder.
2. Download `CLiteRTLM.xcframework.zip` and `CLiteRTLM_mac.xcframework.zip` from Placer, place them into `Prebuilts/`, and unzip both files.
3. Edit `Package.swift` in `LiteRT-LM` to point binary targets to the local `Prebuilts/` folder:
   ```swift
   .binaryTarget(
     name: "CLiteRTLM",
     path: "Prebuilts/CLiteRTLM.xcframework"
   ),
   .binaryTarget(
     name: "CLiteRTLM_mac",
     path: "Prebuilts/CLiteRTLM_mac.xcframework"
   ),
   ```

### Step 5: Add Models by Dragging
Drag and drop your model file(s) into the `iosandmac` folder in the Xcode project navigator:
- **LLM Model**: `gemma-4-E2B-it.litertlm`
- **Embedding Model**: `embedding-gemma-v2.litertlm`

> [!NOTE]
> If needed, compatible model files can be downloaded from Hugging Face:
> - [gemma-4-E2B-it-litert-lm](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm)
> - [embedding-gemma-v2-litert-lm](https://huggingface.co/litert-community/embedding-gemma-v2-litert-lm)

### Step 6: Test Sample App by Changing CPU & GPU
Test the sample app by changing the backend (`cpu()` vs `gpu`) in `ContentView.swift`:

```swift
// In ContentView.swift (both Embedding and LLM sections):
let selectedBackend: Backend = .cpu() // or .gpu
```

- **Simulator**: `cpu()`
- **Physical Phone**: `cpu()`, `gpu`
- **Mac**: `cpu()`, `gpu`

---

## 🧪 Testing Suites Walkthrough

1. **Multimodal Embedding Tab**:
   - Tap **Compute All Modalities (1 Run)** (validates Text, Vision, and Audio).
   - Tap **Run 3-Input Verification** (checks Anchor vs Similar vs Opposite similarity passes).

2. **LLM Generation Tab**:
   - Tap preset prompts or enter custom text.
   - Verify live streaming output, TTFT (ms), and total duration.

---

## 🚀 Final Release Sign-Off
Once testing completes successfully across all platforms:
1. Upload finalized binaries (`CLiteRTLM.xcframework.zip` and `CLiteRTLM_mac.xcframework.zip`) to the GitHub Releases page under the corresponding version tag.
2. Update `Package.swift` in `LiteRT-LM` with the public release URLs and checksums.
