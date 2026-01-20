"""
Bare-Bones Chat Agent for Llama 3.2-1B-Instruct

This is a minimal chat interface that demonstrates:
1. How to load a model without quantization
2. How chat history is maintained and fed back to the model
3. The difference between plain text history and tokenized input

No classes, no fancy features - just the essentials.
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# System prompt - This sets the chatbot's behavior and personality
# Change this to customize how the chatbot responds
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

parser = argparse.ArgumentParser(description="Simple chat agent with summarization/dump helpers")
parser.add_argument("--max-context", type=int, default=None, help="Override model max context")
parser.add_argument("--no-history", action="store_true", help="Disable storing conversation history (use only system prompt and current user message)")
args = parser.parse_args()

print("Loading model (this takes 1-2 minutes)...")

# Load tokenizer (converts text to numbers and vice versa)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in half precision (float16) for efficiency
# Use float16 on GPU, or float32 on CPU if needed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,                        # Use FP16 for efficiency
    device_map="auto",                          # Automatically choose GPU/CPU
    low_cpu_mem_usage=True
)

model.eval()  # Set to evaluation mode (no training)
print(f"âœ“ Model loaded! Using device: {model.device}")
print(f"âœ“ Memory usage: ~2.5 GB (FP16)\n")

# ============================================================================
# CHAT HISTORY - This is stored as PLAIN TEXT (list of dictionaries)
# ============================================================================
# The chat history is a list of messages in this format:
# [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! How can I help?"},
#   {"role": "user", "content": "What's 2+2?"},
#   {"role": "assistant", "content": "2+2 equals 4."}
# ]
#
# This is PLAIN TEXT - humans can read it
# The model CANNOT use this directly - it needs to be tokenized first

chat_history = []

# Add system prompt to history (this persists across the entire conversation)
chat_history.append({
    "role": "system",
    "content": SYSTEM_PROMPT
})

# Token tracking counters
total_input_tokens = 0
total_generated_tokens = 0
# Model / tokenizer max context (if available)
model_max_context = 128000  # Default fallback
if args.max_context:
    try:
        model_max_context = int(args.max_context)
        print(f"[Debug] Using max context: {model_max_context} tokens")
    except Exception:
        pass


def get_token_count(history):
    """Return the token count for a given chat history (tokenized length)."""
    # Use the same chat template as generation but do not add a generation prompt
    ids = tokenizer.apply_chat_template(
        history,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(model.device)
    return int(ids.shape[1])


def summarize_and_truncate(history, keep_recent=5, threshold_ratio=0.8):
    """Summarize older messages and truncate history when token count exceeds threshold.

    - Keeps the system prompt at index 0.
    - Summarize messages 1 through N-keep_recent (keep last keep_recent and system).
    - Replaces them by inserting a single User message containing the summary.
    """
    # Determine max context
    max_ctx = int(model_max_context) if model_max_context else 2048
    threshold = int(max_ctx * threshold_ratio)

    try:
        current_tokens = get_token_count(history)
    except Exception:
        # Fall back if tokenization fails for whatever reason
        current_tokens = 0

    if current_tokens <= threshold:
        return history

    # If there aren't enough messages to summarize, do nothing
    if len(history) <= (keep_recent + 1):
        return history

    # Messages to summarize (exclude system prompt at index 0)
    old_messages = history[1:-keep_recent]
    if not old_messages:
        return history

    # Create a compact text blob to summarize
    convo_text = []
    for m in old_messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        convo_text.append(f"{role}: {content}")
    convo_blob = "\n".join(convo_text)

    # Build a small prompt asking the model to summarize
    summary_prompt = [
        history[0],  # preserve system prompt
        {
            "role": "user",
            "content": (
                "Please provide a concise summary (1-3 sentences) of the following conversation. "
                "Keep only the information necessary for future context.\n\n" + convo_blob
            )
        }
    ]

    # Tokenize and generate a short summary
    input_ids = tokenizer.apply_chat_template(
        summary_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        summary_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    summary_tokens = summary_outputs[0][input_ids.shape[1]:]
    summary_text = tokenizer.decode(summary_tokens, skip_special_tokens=True).strip()

    # Log that summarization occurred and show a short preview
    try:
        gen_len = int(summary_tokens.shape[0])
    except Exception:
        gen_len = None
    try:
        new_token_count = get_token_count([
            history[0],
            {"role": "user", "content": f"[Previous conversation summary: {summary_text}]"},
            *history[-keep_recent:]
        ])
    except Exception:
        new_token_count = None
    preview = (summary_text[:200] + "...") if len(summary_text) > 200 else summary_text
    print(f"[Summarized {len(old_messages)} messages into {gen_len} tokens] Preview: {preview}")
    if new_token_count is not None:
        print(f"[Post-summary token count: {new_token_count}/{max_ctx}]")

    # Construct the new history: system prompt, summary as a user message, then recent messages
    new_history = [
        history[0],
        {"role": "user", "content": f"[Previous conversation summary: {summary_text}]"},
        *history[-keep_recent:]
    ]

    return new_history
# ============================================================================
# CHAT LOOP
# ============================================================================

print("="*70)
print("Chat started! Type 'quit' or 'exit' to end the conversation.")
print("="*70 + "\n")

while True:
    # ========================================================================
    # STEP 1: Get user input (PLAIN TEXT)
    # ========================================================================
    user_input = input("You: ").strip()
    
    # Check for exit commands
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break

    # Debug / utility commands
    if user_input == "/dump":
        print("=== CHAT HISTORY DUMP ===")
        for i, m in enumerate(chat_history):
            content = m.get("content", "")
            preview = content.replace("\n", " ")[:200]
            print(f"{i}: {m.get('role')} - {preview}")
        try:
            total_tokens = get_token_count(chat_history)
            print(f"Total tokens (tokenized): {total_tokens}")
        except Exception as e:
            print("Token count unavailable:", e)
        print("=========================")
        continue

    if user_input == "/dump_full":
        print("=== CHAT HISTORY DUMP ===")
        for i, m in enumerate(chat_history):
            content = m.get("content", "")
            preview = content.replace("\n", " ")
            print(f"{i}: {m.get('role')} - {preview}")
        try:
            total_tokens = get_token_count(chat_history)
            print(f"Total tokens (tokenized): {total_tokens}")
        except Exception as e:
            print("Token count unavailable:", e)
        print("=========================")
        continue

    if user_input == "/force-summarize" or user_input == "/summarize":
        old_len = len(chat_history)
        chat_history = summarize_and_truncate(chat_history)
        if len(chat_history) != old_len:
            print("[Forced summarization applied]")
        else:
            print("[No summarization needed]")
        continue
    
    # Skip empty inputs
    if not user_input:
        continue
    
    # ========================================================================
    # STEP 2: Add user message to chat history (PLAIN TEXT)
    # ========================================================================
    # Build the history used for this turn. If `--no-history` is set,
    # only include the system prompt and the current user message.
    if args.no_history:
        turn_history = [
            chat_history[0],
            {"role": "user", "content": user_input}
        ]
    else:
        # The chat history grows with each exchange
        # We append the new user message to the existing history
        chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Context management: summarize and truncate older messages when tokens get large
        chat_history = summarize_and_truncate(chat_history)
        turn_history = chat_history
    
    # At this point, chat_history looks like:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},      â† Just added
    # ]
    # This is still PLAIN TEXT
    
    # ========================================================================
    # STEP 3: Convert chat history to model input (TOKENIZATION)
    # ========================================================================
    # The model needs numbers (tokens), not text
    # apply_chat_template() does two things:
    #   1. Formats the chat history with special tokens (like <|start|>, <|end|>)
    #   2. Converts the formatted text into token IDs (numbers)
    
    # First, apply_chat_template formats the history and converts to tokens
    input_ids = tokenizer.apply_chat_template(
        turn_history,                    # Our PLAIN TEXT history for this turn
        add_generation_prompt=True,      # Add prompt for assistant's response
        return_tensors="pt"              # Return as PyTorch tensor (numbers)
    ).to(model.device)

    # Create attention mask (1 for all tokens since we have no padding)
    attention_mask = torch.ones_like(input_ids)

    # Count input tokens for this turn
    input_tokens = input_ids.shape[1]
    total_input_tokens += int(input_tokens)

    # Now input_ids is TOKENIZED - it's a tensor of numbers like:
    # tensor([[128000, 128006, 9125, 128007, 271, 2675, 527, 264, ...]])
    # These numbers represent our entire conversation history

    # ========================================================================
    # STEP 4: Generate assistant response (MODEL INFERENCE)
    # ========================================================================
    # The model looks at the ENTIRE chat history (in tokenized form)
    # and generates a response

    print("Assistant: ", end="", flush=True)

    with torch.no_grad():  # Don't calculate gradients (we're not training)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,   # Explicitly pass attention mask
            max_new_tokens=512,              # Maximum length of response
            do_sample=True,                  # Use sampling for variety
            temperature=0.7,                 # Lower = more focused, higher = more random
            top_p=0.9,                       # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )
    
    # outputs contains: [original input tokens + new generated tokens]
    # We only want the NEW tokens (the assistant's response)
    
    # ========================================================================
    # STEP 5: Decode the response (DETOKENIZATION)
    # ========================================================================
    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    generated_tokens = int(new_tokens.shape[0])
    total_generated_tokens += generated_tokens

    # Convert tokens (numbers) back to text (PLAIN TEXT)
    assistant_response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True  # Remove special tokens like <|end|>
    )

    print(assistant_response)  # Display the response

    # Print per-turn token usage stats
    stats = {
        "num_tokens": input_tokens + generated_tokens,
        "max_tokens": model_max_context
    }
    print(f"Using {stats['num_tokens']}/{stats['max_tokens']} tokens")
    
    # ========================================================================
    # STEP 6: Add assistant response to chat history (PLAIN TEXT)
    # ========================================================================
    # This is crucial! We add the assistant's response to the history
    # so the model remembers what it said in future turns
    
    if not args.no_history:
        chat_history.append({
            "role": "assistant",
            "content": assistant_response
        })
    
    # Now chat_history has grown again:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},
    #   {"role": "assistant", "content": "4"}              â† Just added
    # ]
    
    # When the loop repeats:
    # - User enters new message
    # - We add it to chat_history
    # - We tokenize the ENTIRE history (including all previous exchanges)
    # - Model sees everything and generates response
    # - We add response to history
    # - Repeat...
    
    # This is how the chatbot "remembers" the conversation!
    # Each turn, we feed it the ENTIRE conversation history
    
    print()  # Blank line for readability

# ============================================================================
# SUMMARY OF HOW CHAT HISTORY WORKS
# ============================================================================
"""
PLAIN TEXT vs TOKENIZED:

1. PLAIN TEXT (chat_history):
   - Human-readable format
   - List of dictionaries: [{"role": "user", "content": "Hi"}, ...]
   - Stored in memory between turns
   - Gets longer with each message

2. TOKENIZED (input_ids):
   - Numbers (token IDs)
   - Created fresh each turn from chat_history
   - This is what the model actually "reads"
   - Example: [128000, 128006, 9125, 128007, ...]

PROCESS EACH TURN:
   User input (text)
   â†“
   Add to chat_history (text)
   â†“
   Tokenize entire chat_history (text â†’ numbers)
   â†“
   Model generates response (numbers)
   â†“
   Decode response (numbers â†’ text)
   â†“
   Add response to chat_history (text)
   â†“
   Loop back to start

WHY FEED ENTIRE HISTORY?
- The model has no memory between calls
- Each generation is independent
- To "remember" previous turns, we must include them in the input
- This is why context length matters - longer conversations = more tokens

WHAT HAPPENS AS CONVERSATION GROWS?
- chat_history gets longer (more messages)
- Tokenized input gets longer (more tokens)
- Eventually hits model's max context length (for Llama 3.2: 128K tokens)
- Then you need context management (truncation, summarization, etc.)
- But for this simple demo, we let it grow without limit
"""