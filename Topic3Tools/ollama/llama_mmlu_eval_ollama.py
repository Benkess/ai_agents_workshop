"""
Llama 3.2-1B MMLU Evaluation Script using Ollama

This script evaluates Llama 3.2-1B on the MMLU benchmark using Ollama.
Optimized for laptops by running the model locally via Ollama.

Usage:
1. Install Ollama: https://ollama.ai/
2. Pull the model: ollama pull llama3.2:1b
3. Install: pip install requests datasets tqdm
4. Run: python llama_mmlu_eval_ollama.py
"""

import requests
import json
import time
from datasets import load_dataset
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform

def call_ollama(prompt, model="llama3.2:1b", timeout=120, retries=3):
    url = "http://localhost:11434/api/generate"
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                url,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            last_exc = None
            break
        except requests.RequestException as e:
            last_exc = e
            if attempt >= retries:
                raise RuntimeError(f"Request failed after {retries} attempts: {e}") from e
            wait = min(5 * attempt, 30)
            time.sleep(wait)
    if last_exc is not None:
        # Shouldn't reach here because we re-raised above, but keep safe
        raise RuntimeError(f"Request failed: {last_exc}") from last_exc

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"Non-JSON response (status {resp.status_code}): {resp.text!r}")

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {json.dumps(data, indent=2)}")

    # Accept common possible shapes returned by Ollama / proxies
    if isinstance(data, dict):
        for k in ("response", "text", "output"):
            if k in data:
                return data[k]
        if "results" in data and isinstance(data["results"], list) and data["results"]:
            first = data["results"][0]
            if isinstance(first, dict):
                for k in ("content", "text", "response"):
                    if k in first:
                        return first[k]
            return first
        if "responses" in data and data["responses"]:
            return data["responses"][0]
    return json.dumps(data, indent=2)

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

MODEL_NAME = "llama3.2:1b"

# For quick testing, you can reduce this list
MMLU_SUBJECTS = [
    # "abstract_algebra", "anatomy", 
    "astronomy", "business_ethics",
    # "clinical_knowledge", "college_biology", "college_chemistry",
    # "college_computer_science", "college_mathematics", "college_medicine",
    # "college_physics", "computer_security", "conceptual_physics",
    # "econometrics", "electrical_engineering", "elementary_mathematics",
    # "formal_logic", "global_facts", "high_school_biology",
    # "high_school_chemistry", "high_school_computer_science",
    # "high_school_european_history", "high_school_geography",
    # "high_school_government_and_politics", "high_school_macroeconomics",
    # "high_school_mathematics", "high_school_microeconomics",
    # "high_school_physics", "high_school_psychology", "high_school_statistics",
    # "high_school_us_history", "high_school_world_history", "human_aging",
    # "human_sexuality", "international_law", "jurisprudence",
    # "logical_fallacies", "machine_learning", "management", "marketing",
    # "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    # "nutrition", "philosophy", "prehistory", "professional_accounting",
    # "professional_law", "professional_medicine", "professional_psychology",
    # "public_relations", "security_studies", "sociology", "us_foreign_policy",
    # "virology", "world_religions"
]


def check_ollama():
    """Check if Ollama is running and model is available"""
    print("="*70)
    print("Checking Ollama Setup")
    print("="*70)
    
    try:
        # Test with a simple prompt
        response = call_ollama("Hello", model=MODEL_NAME, timeout=10, retries=1)
        print(f"✓ Ollama is running and {MODEL_NAME} is available")
        print(f"✓ Test response: {response.strip()}")
    except RuntimeError as e:
        print(f"❌ Ollama check failed: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is installed: https://ollama.ai/")
        print("2. Ollama is running: ollama serve")
        print(f"3. Model is pulled: ollama pull {MODEL_NAME}")
        sys.exit(1)
    
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")
    print("="*70 + "\n")


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(prompt, model):
    """Get model's prediction for multiple-choice question"""
    response = call_ollama(prompt, model=model)
    
    answer = response.strip()[:1].upper()
    
    if answer not in ["A", "B", "C", "D"]:
        for char in response.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"
    
    return answer


def evaluate_subject(model, subject):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]
        
        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(prompt, model)
        
        if predicted_answer == correct_answer:
            correct += 1
        total += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("Llama 3.2-1B MMLU Evaluation (Ollama)")
    print("="*70 + "\n")

    # Check Ollama setup
    check_ollama()
    
    # Evaluate
    results = []
    total_correct = 0
    total_questions = 0
    
    print(f"\n{'='*70}")
    print(f"Starting evaluation on {len(MMLU_SUBJECTS)} subjects")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
        result = evaluate_subject(MODEL_NAME, subject)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Model: {MODEL_NAME} (via Ollama)")
    print(f"Total Subjects: {len(results)}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Duration: {duration/60:.1f} minutes")
    print("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"llama_3.2_1b_mmlu_results_ollama_{timestamp}.json"
    
    output_data = {
        "model": MODEL_NAME,
        "via": "Ollama",
        "timestamp": timestamp,
        "duration_seconds": duration,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "subject_results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print top/bottom subjects
    if len(results) > 0:
        sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
        
        print("\n📊 Top 5 Subjects:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {result['subject']}: {result['accuracy']:.2f}%")
        
        print("\n📉 Bottom 5 Subjects:")
        for i, result in enumerate(sorted_results[-5:], 1):
            print(f"  {i}. {result['subject']}: {result['accuracy']:.2f}%")
    
    print("\n✅ Evaluation complete!")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
