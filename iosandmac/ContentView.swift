// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import LiteRTLM
import SwiftUI

// MARK: - Main Container View

enum ModelTestTab: String, CaseIterable, Identifiable {
  case embedding = "Multimodal Embedding"
  case llm = "LLM Generation"

  var id: String { rawValue }
}

struct ContentView: View {
  @State private var selectedTab: ModelTestTab = .embedding

  var body: some View {
    VStack(spacing: 0) {
      // Top Navigation Segmented Picker
      Picker("Testing Mode", selection: $selectedTab) {
        ForEach(ModelTestTab.allCases) { tab in
          Text(tab.rawValue).tag(tab)
        }
      }
      .pickerStyle(.segmented)
      .padding()

      Divider()

      // Tab Content
      Group {
        switch selectedTab {
        case .embedding:
          EmbeddingTestingView()
        case .llm:
          LLMTestingView()
        }
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
  }
}

// MARK: - 1. Multimodal Embedding Testing View

struct EmbeddingTestingView: View {
  @State private var statusMessage = "Initializing Embedding Engine..."
  @State private var inputText = "LiteRT high-performance multimodal embeddings on Apple Silicon."
  @State private var logOutput = ""
  @State private var isProcessing = false
  @State private var engine: EmbeddingEngine?

  var body: some View {
    VStack(spacing: 12) {
      Text(statusMessage)
        .font(.headline)
        .multilineTextAlignment(.center)
        .padding(.horizontal)

      TextField("Enter text for embedding", text: $inputText)
        .textFieldStyle(.roundedBorder)
        .padding(.horizontal)

      HStack(spacing: 12) {
        Button("Compute All Modalities (1 Run)") {
          Task {
            await computeAllModalities()
          }
        }
        .buttonStyle(.borderedProminent)
        .disabled(engine == nil || isProcessing)

        Button("Run 3-Input Verification") {
          Task {
            await run3InputVerification()
          }
        }
        .buttonStyle(.bordered)
        .disabled(engine == nil || isProcessing)
      }

      ScrollView {
        Text(logOutput.isEmpty ? "Tap 'Compute All Modalities' or 'Run 3-Input Verification'..." : logOutput)
          .font(.system(.footnote, design: .monospaced))
          .padding()
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      .background(Color.secondary.opacity(0.12))
      .cornerRadius(8)
      .padding(.horizontal)

      Spacer()
    }
    .padding(.vertical)
    .task {
      if engine == nil {
        await initializeEmbeddingEngine()
      }
    }
  }

  // MARK: Sample File Finder (App Bundle)

  func resolveSampleFile(name: String, ext: String, subfolder: String) -> String? {
    if let bundlePath = Bundle.main.path(forResource: name, ofType: ext) {
      return bundlePath
    }
    if let bundleSubPath = Bundle.main.path(forResource: name, ofType: ext, inDirectory: "samples/\(subfolder)") {
      return bundleSubPath
    }
    if let bundleSubPath = Bundle.main.path(forResource: name, ofType: ext, inDirectory: subfolder) {
      return bundleSubPath
    }
    return nil
  }

  // MARK: Engine Initialization

  func initializeEmbeddingEngine() async {
    do {
      guard let modelPath = Bundle.main.path(
        forResource: "embeddinggemma-2-740m", ofType: "litertlm"
      ) ?? Bundle.main.path(
        forResource: "embedding-gemma-v2", ofType: "litertlm"
      ) ?? Bundle.main.path(
        forResource: "model", ofType: "litertlm"
      ) else {
        statusMessage = "❌ Model file (embeddinggemma-2-740m.litertlm) not found in bundle!"
        return
      }

      let cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!

      // Change backend for testing:
      // Simulator:      .cpu()
      // Physical Phone: .cpu(), .gpu
      // Mac:            .cpu(), .gpu
      let selectedBackend: Backend = .cpu() // or .gpu

      let config = EmbeddingEngineConfig(
        modelPath: modelPath,
        backend: selectedBackend,
        visionBackend: selectedBackend,
        audioBackend: selectedBackend,
        cacheDir: cacheDirectory.path,
        maxInputLength: 128,
        visionTokensPerImage: 70
      )

      let newEngine = EmbeddingEngine(config: config)
      try await newEngine.initialize()
      self.engine = newEngine

      let backendName = (selectedBackend == .gpu) ? "Metal GPU" : "CPU"
      statusMessage = "Embedding Engine Ready! 🚀 (\(backendName))"
    } catch {
      statusMessage = "Failed to initialize: \(error.localizedDescription)"
    }
  }

  // MARK: 1 Run for All Modalities

  func computeAllModalities() async {
    guard let engine = engine else { return }
    isProcessing = true
    defer { isProcessing = false }

    logOutput = "--- Single Embedding for Text, Vision & Audio (1 Run) ---\n"

    do {
      // 1. Text
      logOutput += "\n[1] Text Modality:"
      logOutput += "\n    Input: \"\(inputText)\"\n"
      let t0 = CFAbsoluteTimeGetCurrent()
      let textResp = try await engine.computeEmbedding(
        contents: [.text(inputText)],
        options: EmbeddingOptions(normalize: true)
      )
      let textMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
      let textPreview = textResp.embedding.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", ")
      logOutput += "    ✓ Dimension: \(textResp.embedding.count)\n"
      logOutput += "    ✓ Preview:   [\(textPreview), ...]\n"
      logOutput += "    ✓ Latency:   \(String(format: "%.2f", textMs)) ms\n"

      // 2. Vision
      if let imagePath = resolveSampleFile(name: "apple", ext: "png", subfolder: "vision") {
        logOutput += "\n[2] Vision Modality:"
        logOutput += "\n    Input: apple.png\n"
        let t1 = CFAbsoluteTimeGetCurrent()
        let imgResp = try await engine.computeEmbedding(
          contents: [.imageFile(imagePath)],
          options: EmbeddingOptions(normalize: true)
        )
        let imgMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0
        let imgPreview = imgResp.embedding.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        logOutput += "    ✓ Dimension: \(imgResp.embedding.count)\n"
        logOutput += "    ✓ Preview:   [\(imgPreview), ...]\n"
        logOutput += "    ✓ Latency:   \(String(format: "%.2f", imgMs)) ms\n"
      } else {
        logOutput += "\n[2] Vision: ⚠️ apple.png not found\n"
      }

      // 3. Audio
      if let audioPath = resolveSampleFile(name: "have_a_wonderful_day", ext: "wav", subfolder: "audio") {
        logOutput += "\n[3] Audio Modality:"
        logOutput += "\n    Input: have_a_wonderful_day.wav\n"
        let t2 = CFAbsoluteTimeGetCurrent()
        let audResp = try await engine.computeEmbedding(
          contents: [.audioFile(audioPath)],
          options: EmbeddingOptions(normalize: true)
        )
        let audMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000.0
        let audPreview = audResp.embedding.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        logOutput += "    ✓ Dimension: \(audResp.embedding.count)\n"
        logOutput += "    ✓ Preview:   [\(audPreview), ...]\n"
        logOutput += "    ✓ Latency:   \(String(format: "%.2f", audMs)) ms\n"
      } else {
        logOutput += "\n[3] Audio: ⚠️ have_a_wonderful_day.wav not found\n"
      }
    } catch {
      logOutput += "\n❌ Error: \(error.localizedDescription)\n"
    }
  }

  // MARK: 3-Input Verification

  func run3InputVerification() async {
    guard let engine = engine else { return }
    isProcessing = true
    defer { isProcessing = false }

    logOutput = "==================================================\n"
    logOutput += " PART 2: 3-INPUT SIMILARITY VERIFICATION\n"
    logOutput += " (Anchor vs Similar [HIGH] vs Opposite [LOW])\n"
    logOutput += "==================================================\n\n"

    do {
      logOutput += "[Test A] TEXT: 2 Similar vs 1 Opposite\n"
      let t1Anchor = "Apples are delicious and nutritious round fruits."
      let t2Similar = "A crisp, fresh red apple is great for a healthy snack."
      let t3Opposite = "Quantum computers process information using quantum bits and superposition."

      let e1 = try await engine.computeEmbedding(contents: [.text(t1Anchor)], options: EmbeddingOptions(normalize: true)).embedding
      let e2 = try await engine.computeEmbedding(contents: [.text(t2Similar)], options: EmbeddingOptions(normalize: true)).embedding
      let e3 = try await engine.computeEmbedding(contents: [.text(t3Opposite)], options: EmbeddingOptions(normalize: true)).embedding

      let simSame = cosineSimilarity(e1, e2)
      let simOpp = cosineSimilarity(e1, e3)
      let isCorrect = simSame > simOpp

      logOutput += "  Anchor:   \"\(t1Anchor)\"\n"
      logOutput += "  Similar:  \"\(t2Similar)\"\n"
      logOutput += "  Opposite: \"\(t3Opposite)\"\n"
      logOutput += "  --> Sim(Anchor, Similar):  \(String(format: "%.4f", simSame)) (Expected HIGH)\n"
      logOutput += "  --> Sim(Anchor, Opposite): \(String(format: "%.4f", simOpp)) (Expected LOW)\n"
      logOutput += "  Result: \(isCorrect ? "✅ PASS" : "❌ FAIL") (Δ = +\(String(format: "%.4f", simSame - simOpp)))\n\n"

      let appleImg = resolveSampleFile(name: "apple", ext: "png", subfolder: "vision")
      let carImg = resolveSampleFile(name: "blue_car", ext: "png", subfolder: "vision")

      if let applePath = appleImg, let carPath = carImg {
        logOutput += "[Test B] CROSS-MODAL: Text Query vs Image Match vs Image Opposite\n"
        let qApple = "A fresh red apple fruit"
        let eq = try await engine.computeEmbedding(contents: [.text(qApple)], options: EmbeddingOptions(normalize: true)).embedding
        let ev1 = try await engine.computeEmbedding(contents: [.imageFile(applePath)], options: EmbeddingOptions(normalize: true)).embedding
        let ev2 = try await engine.computeEmbedding(contents: [.imageFile(carPath)], options: EmbeddingOptions(normalize: true)).embedding

        let simMatch = cosineSimilarity(eq, ev1)
        let simDiff = cosineSimilarity(eq, ev2)
        logOutput += "  Query: \"\(qApple)\"\n"
        logOutput += "  --> Sim(Query, apple.png):    \(String(format: "%.4f", simMatch))\n"
        logOutput += "  --> Sim(Query, blue_car.png): \(String(format: "%.4f", simDiff))\n"
        logOutput += "  Result: \(simMatch > simDiff ? "✅ PASS" : "❌ FAIL")\n"
      } else {
        logOutput += "[Test B] Vision: ⚠️ Images not found.\n"
      }
    } catch {
      logOutput += "❌ Verification error: \(error.localizedDescription)\n"
    }
  }

  func cosineSimilarity(_ v1: [Float], _ v2: [Float]) -> Float {
    guard v1.count == v2.count, !v1.isEmpty else { return 0 }
    var dot: Float = 0
    var norm1: Float = 0
    var norm2: Float = 0
    for i in 0..<v1.count {
      dot += v1[i] * v2[i]
      norm1 += v1[i] * v1[i]
      norm2 += v2[i] * v2[i]
    }
    let denom = sqrt(norm1) * sqrt(norm2)
    return denom > 0 ? (dot / denom) : 0
  }
}

// MARK: - 2. LLM Generation Testing View

struct LLMTestingView: View {
  @State private var statusMessage = "Initializing LLM Engine..."
  @State private var prompt = "Explain quantum computing in one sentence."
  @State private var responseText = ""
  @State private var isGenerating = false
  @State private var metricsText = ""
  @State private var engine: Engine?
  @State private var conversation: Conversation?

  var body: some View {
    VStack(spacing: 12) {
      Text(statusMessage)
        .font(.headline)
        .multilineTextAlignment(.center)
        .padding(.horizontal)

      // Prompt presets for testing
      ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 8) {
          Text("Presets:").font(.caption).foregroundColor(.secondary)
          Button("One-sentence QC") {
            prompt = "Explain quantum computing in one sentence."
          }
          .buttonStyle(.bordered)
          .controlSize(.small)

          Button("3 Benefits of Swift") {
            prompt = "List 3 key benefits of Swift for Apple Silicon developers."
          }
          .buttonStyle(.bordered)
          .controlSize(.small)

          Button("Reasoning / Math") {
            prompt = "If a train travels 60 miles in 45 minutes, what is its average speed in miles per hour?"
          }
          .buttonStyle(.bordered)
          .controlSize(.small)
        }
        .padding(.horizontal)
      }

      TextField("Enter prompt for LLM...", text: $prompt)
        .textFieldStyle(.roundedBorder)
        .padding(.horizontal)

      HStack(spacing: 12) {
        Button(isGenerating ? "Generating..." : "Generate Response") {
          Task {
            await generateText()
          }
        }
        .buttonStyle(.borderedProminent)
        .disabled(conversation == nil || isGenerating)

        Button("Clear") {
          responseText = ""
          metricsText = ""
        }
        .buttonStyle(.bordered)
        .disabled(responseText.isEmpty)
      }

      if !metricsText.isEmpty {
        Text(metricsText)
          .font(.caption)
          .foregroundColor(.secondary)
      }

      ScrollView {
        Text(responseText.isEmpty ? "Tap 'Generate Response' to test LLM generation..." : responseText)
          .font(.system(.body, design: .rounded))
          .padding()
          .frame(maxWidth: .infinity, alignment: .leading)
      }
      .background(Color.secondary.opacity(0.12))
      .cornerRadius(8)
      .padding(.horizontal)

      Spacer()
    }
    .padding(.vertical)
    .task {
      if engine == nil {
        await initializeLLMEngine()
      }
    }
  }

  // MARK: LLM Engine Initialization

  func initializeLLMEngine() async {
    do {
      guard let modelPath = Bundle.main.path(
        forResource: "gemma-4-E2B-it", ofType: "litertlm"
      ) ?? Bundle.main.path(
        forResource: "model", ofType: "litertlm"
      ) else {
        statusMessage = "❌ LLM model file (gemma-4-E2B-it.litertlm) not found in bundle!"
        return
      }

      let cacheDirectory = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!

      // Change backend for testing:
      // Simulator:      .cpu()
      // Physical Phone: .cpu(), .gpu
      // Mac:            .cpu(), .gpu
      let selectedBackend: Backend = .gpu // or .cpu()

      let config = try EngineConfig(
        modelPath: modelPath,
        backend: selectedBackend,
        cacheDir: cacheDirectory.path
      )
      let newEngine = Engine(engineConfig: config)

      // Offload initialization to background thread to keep UI smooth
      let backendName = (selectedBackend == .gpu) ? "Metal GPU" : "CPU"
      statusMessage = "Loading LLM weights into \(backendName)..."
      try await newEngine.initialize()

      self.engine = newEngine
      self.conversation = try newEngine.createConversation()
      statusMessage = "LLM Engine Ready! 🚀 (\(backendName))"
    } catch {
      statusMessage = "Failed to initialize LLM: \(error.localizedDescription)"
    }
  }

  // MARK: Text Generation with Streaming & Metrics

  func generateText() async {
    guard let conversation = conversation else { return }
    isGenerating = true
    responseText = ""
    metricsText = ""
    defer { isGenerating = false }

    let startTime = CFAbsoluteTimeGetCurrent()
    var firstTokenTime: CFAbsoluteTime?
    var chunkCount = 0

    do {
      for try await chunk in conversation.sendMessageStream(Message(prompt)) {
        if firstTokenTime == nil {
          firstTokenTime = CFAbsoluteTimeGetCurrent()
        }
        chunkCount += 1

        if let firstContent = chunk.contents.first {
          switch firstContent {
          case .text(let text):
            responseText += text
          default:
            break
          }
        }
      }

      let totalDuration = CFAbsoluteTimeGetCurrent() - startTime
      let ttftMs = firstTokenTime.map { ($0 - startTime) * 1000.0 } ?? 0
      metricsText = String(format: "⏱️ TTFT: %.1f ms | Total: %.2f s | Stream chunks: %d", ttftMs, totalDuration, chunkCount)
    } catch {
      responseText = "Inference error: \(error.localizedDescription)"
    }
  }
}
