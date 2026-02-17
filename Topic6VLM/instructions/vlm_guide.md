# Guide to Vision-Language Models (VLMs)

This guide covers the major open-weight and OpenAI vision-language models as of early 2026, including their sizes, computational requirements, and what each version is specialized for.

---

## Open-Weight Models

### LLaVA (Large Language-and-Vision Assistant)

**Developer:** University of Wisconsinâ€“Madison / Microsoft Research  
**License:** Apache 2.0 (LLaVA 1.6); earlier versions restricted by LLaMA license  
**Architecture:** CLIP ViT-L/14 vision encoder + MLP projector + LLM backbone  

| Version | LLM Backbone | Parameters | Ollama Size | Min VRAM | Context |
|---------|-------------|------------|-------------|----------|---------|
| LLaVA 1.5 7B | Vicuna 7B (LLaMA) | ~7.3B total | 4.7 GB | ~6 GB | 32K |
| LLaVA 1.5 13B | Vicuna 13B | ~13.3B total | 8.0 GB | ~10 GB | 4K |
| LLaVA-NeXT 7B | Mistral 7B | ~7.3B total | ~5 GB | ~6 GB | 32K |
| LLaVA-NeXT 13B | Vicuna 13B | ~13.3B total | ~8 GB | ~10 GB | 4K |
| LLaVA-NeXT 34B | Yi-34B | ~34B total | 20 GB | ~24 GB | 4K |

**Version specializations:**
- **LLaVA 1.5** upgraded to 336px image resolution and an MLP projector (from a linear layer), significantly improving visual detail recognition. The 7B variant is the best balance of quality and accessibility for student use.
- **LLaVA-NeXT (1.6)** added dynamic resolution (up to 672Ã—672), improved OCR and text recognition, better visual reasoning, and expanded LLM backbone options. The 34B variant offers the highest quality but requires substantial hardware.

**Best for:** Education and experimentation. Simple architecture makes internals easy to understand. Runs well on student laptops via Ollama. Primarily single-image conversations.

---

### Llama 3.2 Vision (Meta)

**Developer:** Meta  
**License:** Llama 3.2 Community License (commercial use permitted; EU restrictions on multimodal models)  
**Architecture:** Llama 3.1 text model (frozen) + vision adapter with cross-attention layers  

| Version | Text Backbone | Parameters | Min VRAM | Context |
|---------|--------------|------------|----------|---------|
| Llama 3.2 11B Vision | Llama 3.1 8B | 11B | 8 GB | 128K |
| Llama 3.2 90B Vision | Llama 3.1 70B | 90B | 64 GB | 128K |

Both come in **base** and **instruct** variants.

**Key characteristics:**
- The text model parameters are frozen during vision training, preserving all text-only capabilities. This means the 11B model performs identically to Llama 3.1 8B on text tasks.
- Uses a separately trained vision adapter (cross-attention layers), architecturally different from LLaVA's MLP projector approach.
- Trained on 6 billion image-text pairs.
- Works best attending to a single image per conversation, though multi-turn is supported.
- The 11B model is practical for local deployment; the 90B model needs multi-GPU setups.

**Best for:** Drop-in replacement for Llama 3.1 text models when you need vision capabilities. Strong OCR, document understanding, chart interpretation. Available through Ollama.

---

### Qwen 2.5 VL (Alibaba)

**Developer:** Alibaba Cloud / Qwen Team  
**License:** Qwen License (permissive, allows commercial use)  
**Architecture:** ViT with window attention + Qwen 2.5 LLM  

| Version | Parameters | Min VRAM (FP16) | Min VRAM (4-bit) | Context |
|---------|------------|-----------------|------------------|---------|
| Qwen 2.5 VL 3B | 3B | ~8 GB | ~4 GB | 32K |
| Qwen 2.5 VL 7B | 7B | ~17 GB | ~6 GB | 32K |
| Qwen 2.5 VL 72B | 72B | ~144 GB | ~36 GB | 32K (YaRN to 128K) |

**Key characteristics:**
- Dynamic resolution processing â€” adapts to different image sizes and aspect ratios without fixed preprocessing.
- Video input support with temporal frame sampling â€” can process videos at various frame rates and reason about timing.
- Object localization â€” can identify and output bounding box coordinates for objects.
- Strong multilingual support across 29+ languages, including multilingual OCR.
- The 7B model runs on an RTX 4090 (24 GB) with flash attention enabled.
- The 72B model requires multi-GPU setups (e.g., 8Ã—A100) or heavy quantization.

**Best for:** Applications requiring video understanding, multilingual OCR, or object localization. The most feature-rich open VLM. The 7B variant offers an excellent quality-to-cost ratio.

---

### Gemma 3 (Google)

**Developer:** Google DeepMind  
**License:** Gemma License (permissive, commercial use permitted)  
**Architecture:** Decoder-only transformer + frozen SigLIP vision encoder (~400M params)  

| Version | Parameters | Vision Support | Min VRAM (FP16) | Min VRAM (QAT/4-bit) | Context |
|---------|------------|---------------|-----------------|----------------------|---------|
| Gemma 3 270M | 270M | No | <1 GB | <1 GB | 32K |
| Gemma 3 1B | 1B | No | ~2 GB | ~1 GB | 32K |
| Gemma 3 4B | 4B | Yes | ~8 GB | ~3 GB | 128K |
| Gemma 3 12B | 12B | Yes | ~24 GB | ~8 GB | 128K |
| Gemma 3 27B | 27B | Yes | ~54 GB | ~18 GB | 128K |

**Key characteristics:**
- The same frozen SigLIP vision encoder is shared across the 4B, 12B, and 27B models (270M and 1B are text-only).
- "Pan & Scan" algorithm handles high-resolution and non-square images by adaptively cropping into 896Ã—896 patches.
- Bidirectional attention for image tokens (unlike text, which uses standard causal attention).
- Interleaved local/global attention (5:1 ratio) dramatically reduces KV-cache memory for long contexts.
- Quantization-Aware Training (QAT) checkpoints provided, preserving quality at ~3Ã— less memory.
- 140+ language support.
- The 27B QAT version can run on a consumer RTX 3090 (24 GB).

**Best for:** Long-context multimodal tasks (128K tokens â‰ˆ 500 images or 8+ minutes of video at 1fps). Best open model for running on consumer hardware when using QAT variants. Strong multilingual capabilities.

---

### Phi-4 Multimodal (Microsoft)

**Developer:** Microsoft  
**License:** MIT  
**Architecture:** Unified transformer with Mixture-of-LoRAs for text, vision, and speech  

| Version | Parameters | Min VRAM | Context |
|---------|------------|----------|---------|
| Phi-4-mini | 3.8B (text only) | ~4 GB | 128K |
| Phi-4-multimodal | 5.6B (text + vision + audio) | ~8 GB | 128K |
| Phi-4 | 14B (text only) | ~10 GB (FP16) | 16K |

**Key characteristics:**
- Phi-4-multimodal is a true omni-model: a single architecture that handles text, images, *and* audio simultaneously. Not a pipeline of separate models.
- Uses Mixture-of-LoRAs to handle different modalities while minimizing interference between them.
- Remarkably capable for its size â€” competes with models 2â€“3Ã— larger on vision benchmarks, especially in mathematical and scientific reasoning with images.
- Strong OCR, chart analysis, and document understanding.
- Designed for edge/on-device deployment.
- Holds #1 on the HuggingFace OpenASR leaderboard for speech recognition (6.14% word error rate).

**Best for:** Edge/mobile deployment where you need vision + audio in a tiny package. Excellent for educational settings due to low hardware requirements. Best bang-for-parameter on STEM visual reasoning.

---

### DeepSeek-VL2 (DeepSeek)

**Developer:** DeepSeek AI  
**License:** DeepSeek License (permissive)  
**Architecture:** Mixture-of-Experts (MoE) with vision encoder  

| Version | Total Params | Active Params | Min VRAM |
|---------|-------------|---------------|----------|
| DeepSeek-VL2-Tiny | 3.4B | 1.0B | ~4 GB |
| DeepSeek-VL2-Small | 16.1B | 2.8B | ~8 GB |
| DeepSeek-VL2 | 27.5B | 4.5B | ~12 GB |

**Key characteristics:**
- MoE architecture means only a fraction of parameters are active per token, making it very efficient at inference despite large total parameter counts.
- Strong scientific and mathematical reasoning.
- Dynamic tiling for high-resolution image processing.
- The Tiny variant (1B active params) is one of the smallest capable VLMs available.

**Best for:** Cost-efficient inference where you need more capability than parameter count suggests. Good for scientific/technical image analysis.

---

### Other Notable Open Models

**InternVL 2.5** (Shanghai AI Lab): Available from 1B to 78B. Very strong benchmark performance, especially at larger sizes. Uses InternViT vision encoder.

**MiniCPM-o 2.6** (OpenBMB): 8B parameters. Uniquely supports images, video, *and* audio input. Real-time speech conversation capability. Good all-rounder for multimodal tasks.

**Janus-Pro** (DeepSeek): Available in 1.5B and 7B. Can both *understand* and *generate* images (bidirectional), unlike most VLMs which only understand images.

---

## OpenAI Models

OpenAI's models are proprietary and accessed only via API (or ChatGPT). Parameter counts are not publicly disclosed. All support vision (image input).

### GPT-4o

**Released:** May 2024  
**Context Window:** 128K tokens  

| Variant | Best For | Relative Cost | Relative Speed |
|---------|---------|---------------|----------------|
| GPT-4o | General-purpose multimodal (text + vision + audio) | Medium | Fast |
| GPT-4o-mini | Budget-friendly general use | Low | Very fast |

**Key characteristics:**
- Natively multimodal â€” single model trained end-to-end on text, vision, and audio (not a pipeline).
- Audio response latency as low as 232ms, enabling real-time voice conversation.
- Strong across all modalities but not specialized for any one task.
- Available in ChatGPT (free and paid tiers) and via API.

**Best for:** General-purpose multimodal applications. Real-time voice + vision interactions. Accessible default choice for most users.

---

### GPT-4.1

**Released:** April 2025  
**Context Window:** 1 million tokens  

| Variant | Best For | Relative Cost | Relative Speed |
|---------|---------|---------------|----------------|
| GPT-4.1 | Coding, long-context, agentic tasks, vision | High | Medium |
| GPT-4.1 mini | Same strengths, lower cost | Low | Fast |
| GPT-4.1 nano | Classification, autocomplete, fast tasks | Very low | Very fast |

**Key characteristics:**
- Major improvements in coding (55% on SWE-bench, +21.4 points over GPT-4o) and instruction following.
- 1M token context window across all three variants.
- Strong vision performance â€” tested on MMMU (image Q&A), MathVista (visual math), CharXiv (chart analysis).
- Set state-of-the-art on Video-MME benchmark for multimodal long-context video understanding.
- Optimized for tool calling and agentic workflows.
- GPT-4.1 mini matches or exceeds GPT-4o on many benchmarks at 83% lower cost and nearly half the latency.

**Best for:** Developer/agentic applications requiring precise instruction following, long documents with images, code generation from visual specs. The mini variant is the practical sweet spot for most API use cases.

---

### GPT-5

**Released:** August 2025  
**Context Window:** 400K tokens  

**Key characteristics:**
- Not a single model but a routing system: automatically directs queries to either a fast model (gpt-5-main, successor to GPT-4o) or a reasoning model (gpt-5-thinking, successor to o-series) based on task complexity.
- Represents OpenAI's current flagship â€” highest overall capability.
- Available in ChatGPT and via API.

**Best for:** Highest-capability tasks where you want OpenAI to automatically select the right model for the job. Premium pricing.

---

## Quick Comparison for Classroom Use

For an educational setting with limited budgets and student laptops, here are practical recommendations:

| Goal | Recommended Model | Why |
|------|------------------|-----|
| Simplest local setup | LLaVA 7B via Ollama | 4.7 GB download, runs on most laptops, simple API |
| Best local quality | Gemma 3 12B QAT via Ollama | Good quality at ~8 GB VRAM with quantization |
| Smallest capable model | Phi-4-multimodal (5.6B) | Vision + audio in one tiny model |
| Best free cloud API | Google Gemini Flash (free tier) | No local hardware needed, fast |
| Understanding internals | LLaVA via HuggingFace Transformers | Components (CLIP, projector, LLM) are accessible individually |
| Multi-image / video | Qwen 2.5 VL 7B | Native video support, dynamic resolution |
| Comparing open vs. proprietary | GPT-4.1 mini via API | Low cost, strong benchmarks, good baseline for comparison |

---

## Computational Requirements Summary

**Rule of thumb for VRAM estimation:**  
- FP16/BF16: ~2 GB per billion parameters  
- 8-bit quantized: ~1 GB per billion parameters  
- 4-bit quantized: ~0.5 GB per billion parameters  
- Add 20â€“30% overhead for inference buffers and image processing  

**Minimum hardware tiers:**

| Hardware | Can Run |
|----------|---------|
| 8 GB VRAM (e.g., laptop GPU, M1/M2 Mac) | LLaVA 7B, Llama 3.2 11B (quantized), Phi-4-multimodal |
| 16 GB VRAM (e.g., RTX 4060 Ti, M2 Pro Mac) | Most 7B models at FP16, 12â€“13B quantized |
| 24 GB VRAM (e.g., RTX 4090) | 7B at FP16 comfortably, Gemma 3 27B with QAT |
| 48+ GB VRAM (e.g., A6000, dual GPU) | 34B models, larger Qwen variants |
| 80+ GB VRAM (e.g., A100) | 70B+ models |
| Multi-GPU (e.g., 4â€“8Ã— A100) | Qwen 72B, Llama 90B |

**Note:** Apple Silicon Macs (M1/M2/M3/M4) with unified memory can run larger models than their VRAM equivalent would suggest on discrete GPUs, since they share system RAM. An M2 Max with 32 GB can comfortably run 13B models.