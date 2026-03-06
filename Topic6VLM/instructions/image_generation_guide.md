# Guide to Image Generation Models

This guide covers the major open-weight and OpenAI text-to-image generation models as of early 2026, including their sizes, computational requirements, and what each version is specialized for.

---

## How Image Generation Models Work

Unlike VLMs (which take images in and produce text out), image generation models take text in and produce images out. The dominant approaches are:

- **Diffusion models** â€” Start with random noise and iteratively denoise it, guided by a text embedding, until a coherent image emerges. Most run in a compressed "latent space" (via a VAE) for efficiency rather than operating directly on pixels.
- **Flow matching** â€” A mathematically cleaner variant of diffusion that learns straight-path transformations from noise to image, requiring fewer sampling steps.
- **Autoregressive models** â€” Treat image generation like next-token prediction, generating image tokens sequentially (similar to how LLMs generate text).

All approaches use a **text encoder** (typically CLIP, T5, or more recently Mistral/Qwen) to convert the prompt into an embedding that guides generation.

---

## Open-Weight Models

### FLUX (Black Forest Labs)

**Developer:** Black Forest Labs (founded by the original creators of Stable Diffusion)  
**Architecture:** Diffusion Transformer (DiT) + flow matching  

#### FLUX.1 Family

| Variant | Parameters | License | Min VRAM (FP16) | Min VRAM (FP8/quantized) | Speed |
|---------|------------|---------|-----------------|-------------------------|-------|
| FLUX.1 [schnell] | 12B | Apache 2.0 | ~33 GB | ~16 GB (FP8) | Very fast (1â€“4 steps) |
| FLUX.1 [dev] | 12B | Non-commercial | ~22 GB | ~12 GB (FP8) | Medium (20â€“50 steps) |
| FLUX.1 [pro] | 12B | API only | N/A (cloud) | N/A | Medium |

**FLUX.1 specializations:**
- **[schnell]** (German for "fast") â€” Distilled for speed. Generates images in 1â€“4 steps. Fully open-source (Apache 2.0). Best for rapid prototyping and real-time applications.
- **[dev]** â€” The full-quality open-weight model for development and fine-tuning. ~20 steps for best results. Non-commercial license.
- **[pro]** â€” Highest quality, API-only access. Best prompt adherence and visual fidelity.

#### FLUX.2 Family (November 2025)

| Variant | Parameters | License | Min VRAM (FP16) | Min VRAM (FP8/quantized) | Notes |
|---------|------------|---------|-----------------|-------------------------|-------|
| FLUX.2 [dev] | 32B | Non-commercial | ~64 GB | ~18â€“20 GB (4-bit) | Open weights |
| FLUX.2 [pro] | 32B | API only | N/A | N/A | Highest quality |
| FLUX.2 [flex] | 32B | API only | N/A | N/A | Fine-grained control |
| FLUX.2 [klein] 9B | 9B | Non-commercial | ~20 GB | ~10 GB | Fast, unified gen+edit |
| FLUX.2 [klein] 4B | 4B | Apache 2.0 | ~13 GB | ~6 GB | Consumer GPU friendly |

**FLUX.2 improvements over FLUX.1:**
- 32B parameters (up from 12B) for the main models, dramatically improving quality.
- Uses Mistral Small as the text encoder (replacing CLIP), giving much better prompt understanding for complex multi-clause descriptions.
- Up to 4 megapixel (2048Ã—2048) resolution output.
- Clean, legible text rendering in images.
- Multi-reference generation â€” can take up to 10 reference images for consistent character/product generation.
- **[klein]** variants unify generation and editing in one model, running in under a second on consumer GPUs.

**Best for:** FLUX is currently the leading open-weight image generation family. FLUX.1 [schnell] for fast/free generation; FLUX.2 [klein] 4B for consumer-grade real-time use; FLUX.2 [dev] for highest open-weight quality.

---

### Stable Diffusion (Stability AI)

**Developer:** Stability AI  
**Architecture:** Latent diffusion (U-Net or DiT based, depending on version) + VAE  

| Version | Parameters | Architecture | Min VRAM | Resolution | License |
|---------|------------|-------------|----------|-----------|---------|
| SD 1.5 | ~860M | U-Net | ~4 GB | 512Ã—512 | CreativeML Open RAIL-M |
| SDXL | ~3.5B (base + refiner) | U-Net | ~8 GB | 1024Ã—1024 | Open RAIL++ |
| SDXL Turbo | ~3.5B | Distilled U-Net | ~8 GB | 512Ã—512 | Research only |
| SDXL Lightning | ~3.5B | Distilled U-Net | ~8 GB | 1024Ã—1024 | Open RAIL++ |
| SD 3 Medium | 2B | DiT (MMDiT) | ~6 GB | 1024Ã—1024 | Community license |
| SD 3.5 Large | 8B | DiT (MMDiT) | ~12 GB | 1024Ã—1024 | Community license |
| SD 3.5 Large Turbo | 8B | Distilled DiT | ~12 GB | 1024Ã—1024 | Community license |

**Version specializations:**
- **SD 1.5** â€” The original workhorse. Tiny by modern standards but has the largest ecosystem of fine-tuned models, LoRAs, and ControlNets ever built. Still useful for specialized styles via community fine-tunes.
- **SDXL** â€” Major quality upgrade to 1024px native resolution. Two-stage pipeline (base + refiner). The current practical standard for many workflows.
- **SDXL Turbo / Lightning** â€” Distilled versions of SDXL that generate in 1â€“4 steps instead of 20â€“50. Turbo is research-only; Lightning is more permissive. Near-instant generation.
- **SD 3 / 3.5** â€” Architectural shift from U-Net to Multimodal Diffusion Transformer (MMDiT). Uses three text encoders (CLIP Ã—2 + T5). Much better text rendering in images. The "Large" 8B variant offers the best quality.
- **SD 3.5 Large Turbo** â€” Distilled SD 3.5 for fast generation with the new architecture.

**Best for:** Stable Diffusion has the most mature ecosystem (AUTOMATIC1111, ComfyUI, thousands of fine-tunes and LoRAs). SD 1.5 and SDXL remain practical for anyone with existing workflows. SD 3.5 is the best choice for new projects needing text-in-image capability. The small model sizes make Stable Diffusion the most accessible family for limited hardware.

---

### DeepFloyd IF (Stability AI / DeepFloyd)

**Developer:** DeepFloyd (Stability AI research lab)  
**Architecture:** Cascaded pixel-space diffusion (3 stages) + frozen T5-XXL text encoder  
**License:** Research only  

| Stage | Resolution | Parameters |
|-------|-----------|------------|
| Stage 1 | 64Ã—64 | ~4.3B |
| Stage 2 (upscaler) | 256Ã—256 | ~1.2B |
| Stage 3 (upscaler) | 1024Ã—1024 | ~1.2B |
| T5-XXL text encoder | â€” | ~4.7B |

**Total VRAM:** ~40 GB for the full pipeline (can be run stage-by-stage with ~16 GB)

**Key characteristics:**
- Operates in pixel space (not latent space), which was unusual at the time of release.
- Uses T5-XXL as the text encoder, giving it exceptional prompt understanding and text rendering â€” this was groundbreaking before SD 3 and FLUX adopted similar approaches.
- Three-stage cascaded generation: low-res â†’ medium-res â†’ high-res.
- Research license only; not commercially usable.

**Best for:** Academic research and understanding cascaded diffusion architectures. Largely superseded by FLUX and SD 3.5 for practical use.

---

### Community & Emerging Open Models

**Z-Image-Turbo** â€” 6B parameters, leading HuggingFace downloads. Strong quality at a fraction of the size of FLUX.2. Good balance of speed and quality for consumer hardware.

**LongCat-Image** (Meituan) â€” 6B parameters. Bilingual Chinese-English text rendering. Strong photorealism. Has separate dev (for LoRA training) and edit (for image editing) variants.

**Ovis-Image** â€” 7B parameters. Specialized for high-quality text rendering in generated images (posters, logos, UI mockups). Less suited for photorealism.

**Janus-Pro** (DeepSeek) â€” Unique in that it can both *understand* and *generate* images using a unified architecture. Available in 1.5B and 7B sizes. Uses a VQ tokenizer for autoregressive image generation rather than diffusion.

**Playground 2.5** â€” Based on SDXL, trained to mimic Midjourney's aesthetic. Produces polished, detailed images without complex prompting. Good for users who want a "just works" artistic style.

---

## OpenAI Models

OpenAI's image generation models are proprietary and API-only. No local deployment is possible. Parameter counts are not disclosed.

### GPT Image Family (Current)

| Model | Released | Price per Image (1024Ã—1024) | Resolutions | Key Strength |
|-------|---------|---------------------------|-------------|-------------|
| gpt-image-1.5 | 2025 | ~$0.02â€“$0.19 (lowâ€“high quality) | 1024Â², 1024Ã—1536, 1536Ã—1024 | State-of-the-art quality, best text rendering |
| gpt-image-1 | April 2025 | ~$0.02â€“$0.19 (lowâ€“high quality) | 1024Â², 1024Ã—1536, 1536Ã—1024 | Professional grade, streaming support |
| gpt-image-1-mini | 2025 | Lower than gpt-image-1 | 1024Â² | Cost-effective, lower quality |

**Key characteristics:**
- Natively multimodal â€” built into the GPT architecture, not a separate model.
- Supports image *editing* (with masks) in addition to generation.
- Excellent text rendering, world knowledge, and instruction following.
- Streaming support â€” can return partial images as they generate.
- Transparent background support (PNG output).
- Token-based pricing (not flat per-image like DALL-E 3 was).
- C2PA metadata embedded for provenance tracking.

### DALL-E 3 (Deprecated)

| Model | Price per Image | Resolutions |
|-------|----------------|-------------|
| DALL-E 3 Standard | $0.040 (1024Â²), $0.080 (1792Ã—1024) | 1024Â², 1024Ã—1792, 1792Ã—1024 |
| DALL-E 3 HD | $0.080 (1024Â²), $0.120 (1792Ã—1024) | 1024Â², 1024Ã—1792, 1792Ã—1024 |

**Note:** DALL-E 3 is deprecated and will be removed from the API on May 12, 2026. It was replaced by GPT Image models in ChatGPT in March 2025. DALL-E 3 used a diffusion architecture and was notable for its strong prompt adherence (it automatically expanded brief prompts into detailed descriptions via an internal ChatGPT rewrite).

**Best for:** OpenAI's GPT Image models are the easiest way to get high-quality image generation via API. No hardware requirements, no setup. Best text rendering and prompt understanding of any available system. Cost adds up at scale.

---

## Key Technical Concepts

### Guidance Scale (CFG)
Controls how closely the output follows the text prompt. Higher values = more literal prompt following but potentially less natural-looking. Typical range: 5â€“15 for most models.

### Sampling Steps
The number of denoising iterations. More steps = higher quality but slower. Typical range: 20â€“50 for standard models, 1â€“4 for distilled/turbo variants.

### LoRA (Low-Rank Adaptation)
A lightweight fine-tuning technique that trains a small adapter (typically 10â€“100 MB) to customize a model's style or teach it new concepts. Widely used with Stable Diffusion and FLUX. Much cheaper than full fine-tuning.

### ControlNet
An auxiliary network that adds spatial conditioning to diffusion models â€” you can guide generation using edge maps, depth maps, pose skeletons, etc. Available for SD 1.5, SDXL, and FLUX.

### VAE (Variational Autoencoder)
Compresses images to/from latent space. All latent diffusion models (SD, FLUX) use a VAE. The diffusion process runs in the compressed latent space, then the VAE decoder converts back to pixels.

### Text Encoders
The component that converts your text prompt into embeddings that guide generation:
- **CLIP** â€” Used by SD 1.5, SDXL, FLUX.1. Good general understanding but limited on long/complex prompts.
- **T5** â€” Used by SD 3.5, DeepFloyd IF, alongside CLIP. Much better at understanding long, detailed prompts and rendering text.
- **Mistral/Qwen** â€” Used by FLUX.2. Full LLM-grade language understanding for prompts.

---

## Computational Requirements Summary

### Quick VRAM Reference

| Hardware | Can Run |
|----------|---------|
| 4 GB VRAM | SD 1.5 (basic) |
| 6â€“8 GB VRAM | SD 1.5, SDXL (with optimizations), SD 3 Medium, FLUX.2 [klein] 4B (quantized) |
| 12â€“16 GB VRAM | SDXL comfortably, SD 3.5 Large, FLUX.1 [schnell] (FP8), FLUX.2 [klein] 4B |
| 24 GB VRAM (RTX 4090) | FLUX.1 [dev] (FP16), FLUX.2 [dev] (4-bit quantized), all SD variants |
| 48+ GB VRAM | FLUX.2 [dev] (FP8), full pipeline with ControlNets |
| 64+ GB VRAM | FLUX.2 [dev] (FP16) |

### Generation Speed (approximate, RTX 4090, 1024Ã—1024)

| Model | Steps | Time |
|-------|-------|------|
| SD 1.5 | 20 | ~2 seconds |
| SDXL | 20 | ~4 seconds |
| SDXL Lightning | 4 | <1 second |
| SD 3.5 Large | 28 | ~6 seconds |
| FLUX.1 [schnell] | 4 | ~3 seconds |
| FLUX.1 [dev] | 20 | ~10 seconds |
| FLUX.2 [klein] 4B | 4 | <1 second |
| FLUX.2 [dev] (FP8) | 20 | ~15 seconds |

---

## Quick Comparison for Classroom Use

| Goal | Recommended Model | Why |
|------|------------------|-----|
| Smallest/fastest local setup | SD 1.5 or SD 3 Medium | Runs on 4â€“6 GB VRAM, huge ecosystem |
| Best free & open quality | FLUX.1 [schnell] | Apache 2.0, very fast, 12B params |
| Consumer GPU, high quality | FLUX.2 [klein] 4B | Apache 2.0, <1 second, ~13 GB VRAM |
| Understanding diffusion concepts | SD 1.5 via diffusers library | Simplest architecture, most tutorials available |
| No local hardware needed | OpenAI GPT Image API | API-only, ~$0.02â€“$0.19/image |
| Fine-tuning / LoRA training | SDXL or FLUX.1 [dev] | Best ecosystem for LoRA and ControlNet |
| Text rendering in images | SD 3.5 Large or FLUX.2 | Both handle text well due to T5/Mistral encoders |
| Image editing | FLUX.2 [klein] or GPT Image API | Both support unified generation + editing |

---

## Architecture Comparison

| Feature | SD 1.5 / SDXL | SD 3 / 3.5 | FLUX.1 | FLUX.2 |
|---------|---------------|------------|--------|--------|
| Denoising backbone | U-Net | MMDiT (Transformer) | DiT (Transformer) | DiT (Transformer) |
| Text encoder(s) | CLIP | CLIP Ã—2 + T5 | CLIP + T5 | Mistral Small |
| Latent space | Yes (VAE) | Yes (VAE) | Yes (VAE) | Yes (VAE) |
| Approach | Diffusion | Diffusion | Flow matching | Flow matching |
| Text rendering | Poor (SD 1.5), Fair (SDXL) | Good | Good | Excellent |
| Native resolution | 512 (1.5) / 1024 (XL) | 1024 | 1024 | Up to 2048 (4MP) |
| Typical parameters | 0.8Bâ€“3.5B | 2Bâ€“8B | 12B | 4Bâ€“32B |

The clear trend is toward transformer-based backbones (replacing U-Nets), flow matching (replacing classical diffusion), and LLM-grade text encoders (replacing CLIP alone). Each generation roughly doubles in parameter count while improving quality, text understanding, and generation speed through distillation.