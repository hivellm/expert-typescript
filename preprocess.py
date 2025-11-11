#!/usr/bin/env python3
"""
Preprocess TypeScript datasets for Qwen3 training.

This script:
1. Converts legacy ChatML format (<|system|>, <|user|>, <|assistant|>, <|end|>) to Qwen3 native format (<|im_start|>/<|im_end|>)
2. Adds reasoning blocks to 75% of examples (Qwen3 hybrid reasoning)
3. Validates and filters examples
4. Outputs train.jsonl in Qwen3 format
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import argparse
import sys

# Add common preprocessing utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from common_preprocessing_utils import sanitize_chatml_response, extract_query_only


def generate_brief_reasoning(instruction: str, output: str) -> str:
    """Generate a brief reasoning statement for Qwen3 compatibility.
    
    Qwen3 uses hybrid reasoning, so we include concise reasoning that leads to the TypeScript code.
    This helps the model understand when to use reasoning vs direct output.
    """
    # Detect what the TypeScript code is doing
    output_lower = output.lower()
    
    if 'function' in output_lower or 'const' in output_lower or 'let' in output_lower:
        if 'class' in output_lower:
            reasoning = f"I need to create a TypeScript class with the required functionality."
        elif 'interface' in output_lower or 'type' in output_lower:
            reasoning = f"I need to define TypeScript types or interfaces."
        elif 'async' in output_lower or 'await' in output_lower:
            reasoning = f"I need to implement an asynchronous TypeScript function."
        else:
            reasoning = f"I need to implement a TypeScript function or module."
    elif 'import' in output_lower or 'export' in output_lower:
        reasoning = f"I need to set up TypeScript imports and exports."
    else:
        reasoning = f"I need to generate TypeScript code to fulfill the request."
    
    return reasoning


def convert_legacy_to_qwen3(text: str) -> Optional[Tuple[str, str, str]]:
    """Convert legacy ChatML format to Qwen3 format and extract sections.
    
    Legacy format:
    <|system|>\nDialect: typescript\n<|end|>
    <|user|>\n{instruction}\n<|end|>
    <|assistant|>\n{output}\n<|end|>
    
    Qwen3 format:
    <|im_start|>system\nDialect: typescript<|im_end|>
    <|im_start|>user\n{instruction}<|im_end|>
    <|im_start|>assistant\n{output}<|im_end|>
    
    Returns:
        Tuple of (system, user, assistant) content or None if conversion fails
    """
    # Extract system content
    system_match = re.search(r'<\|system\|>\s*\n(.*?)<\|end\|>', text, re.DOTALL)
    if not system_match:
        return None
    system_content = system_match.group(1).strip()
    
    # Extract user content
    user_match = re.search(r'<\|user\|>\s*\n(.*?)<\|end\|>', text, re.DOTALL)
    if not user_match:
        return None
    user_content = user_match.group(1).strip()
    
    # Extract assistant content
    assistant_match = re.search(r'<\|assistant\|>\s*\n(.*?)<\|end\|>', text, re.DOTALL)
    if not assistant_match:
        return None
    assistant_content = assistant_match.group(1).strip()
    
    return (system_content, user_content, assistant_content)


def format_qwen3(
    system: str,
    user: str,
    assistant: str,
    include_reasoning: bool = False
) -> str:
    """Format example with Qwen3 native format (<|im_start|>/<|im_end|>).
    
    Args:
        system: System message content
        user: User message content
        assistant: Assistant response (TypeScript code)
        include_reasoning: If True, wraps assistant content in <think> block for Qwen3 compatibility.
                          Qwen3 uses hybrid reasoning, so some examples should include reasoning blocks.
    """
    # CRITICAL: Sanitize TypeScript to ensure code-only (no reasoning/explanation)
    assistant_clean = sanitize_chatml_response(assistant, query_type="auto")
    if not assistant_clean:
        assistant_clean = extract_query_only(assistant, query_type="auto")
    
    # For Qwen3 compatibility: optionally wrap in reasoning block
    # Qwen3 uses hybrid reasoning: 75% reasoning + 25% direct (as per Qwen3 training notebook)
    if include_reasoning:
        # Generate a brief reasoning that leads to the TypeScript code
        reasoning = generate_brief_reasoning(user, assistant_clean)
        assistant_content = f"<think>\n{reasoning}\n</think>\n{assistant_clean}"
    else:
        assistant_content = assistant_clean
    
    # Qwen3 format: <|im_start|>role\ncontent<|im_end|>
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>\n"
    )


def validate_example(text: str) -> bool:
    """Validate Qwen3 format example."""
    required_tags = ["<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant", "<|im_end|>"]
    
    for tag in required_tags:
        if tag not in text:
            return False
    
    # Check that assistant content is not empty
    assistant_match = re.search(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.DOTALL)
    if not assistant_match or not assistant_match.group(1).strip():
        return False
    
    return True


def process_dataset(
    input_files: List[Path],
    output_file: Path,
    reasoning_ratio: float = 0.75
) -> Dict[str, Any]:
    """Process TypeScript datasets and convert to Qwen3 format with reasoning.
    
    Args:
        input_files: List of input JSONL files
        output_file: Output JSONL file path
        reasoning_ratio: Ratio of examples to include reasoning (default: 0.75 for 75%)
    
    Returns:
        Statistics dictionary
    """
    all_examples = []
    stats = {
        "total_loaded": 0,
        "converted": 0,
        "with_reasoning": 0,
        "direct_output": 0,
        "invalid": 0,
        "skipped": 0
    }
    
    reasoning_counter = 0  # Counter for reasoning distribution
    
    # Load all examples from input files
    print("="*80)
    print("Loading datasets...")
    print("="*80)
    
    for input_file in input_files:
        if not input_file.exists():
            print(f"[WARNING] File not found: {input_file}")
            continue
        
        print(f"\nLoading {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    example = json.loads(line.strip())
                    text = example.get("text", "")
                    
                    if not text:
                        stats["skipped"] += 1
                        continue
                    
                    stats["total_loaded"] += 1
                    
                    # Convert legacy format to Qwen3 format
                    sections = convert_legacy_to_qwen3(text)
                    if not sections:
                        # Try to parse as already Qwen3 format
                        system_match = re.search(r'<\|im_start\|>system\n(.*?)<\|im_end\|>', text, re.DOTALL)
                        user_match = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', text, re.DOTALL)
                        assistant_match = re.search(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', text, re.DOTALL)
                        
                        if system_match and user_match and assistant_match:
                            sections = (
                                system_match.group(1).strip(),
                                user_match.group(1).strip(),
                                assistant_match.group(1).strip()
                            )
                        else:
                            stats["invalid"] += 1
                            continue
                    
                    system, user, assistant = sections
                    
                    # Apply reasoning distribution (75% reasoning + 25% direct)
                    # Qwen3 uses hybrid reasoning: 75% reasoning + 25% direct (as per Qwen3 training notebook)
                    include_reasoning = (reasoning_counter % 4 != 0)  # 75% with reasoning (3 out of 4)
                    reasoning_counter += 1
                    
                    # Format with Qwen3
                    formatted_text = format_qwen3(system, user, assistant, include_reasoning=include_reasoning)
                    
                    # Validate
                    if not validate_example(formatted_text):
                        stats["invalid"] += 1
                        continue
                    
                    all_examples.append({"text": formatted_text})
                    stats["converted"] += 1
                    
                    if include_reasoning:
                        stats["with_reasoning"] += 1
                    else:
                        stats["direct_output"] += 1
                    
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Invalid JSON at line {line_num} in {input_file}: {e}")
                    stats["skipped"] += 1
                    continue
                except Exception as e:
                    print(f"[WARNING] Error processing line {line_num} in {input_file}: {e}")
                    stats["skipped"] += 1
                    continue
    
    # Deduplicate
    print("\n" + "="*80)
    print("Deduplicating...")
    print("="*80)
    
    seen_texts = set()
    unique_examples = []
    
    for example in tqdm(all_examples, desc="Deduplicating"):
        text = example["text"]
        # Use first 500 chars as signature (faster than full hash)
        signature = text[:500]
        
        if signature not in seen_texts:
            seen_texts.add(signature)
            unique_examples.append(example)
    
    stats["duplicates_removed"] = len(all_examples) - len(unique_examples)
    stats["final_count"] = len(unique_examples)
    
    # Save output
    print("\n" + "="*80)
    print(f"Saving to {output_file}...")
    print("="*80)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(unique_examples, desc="Writing"):
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Preprocess TypeScript datasets for Qwen3 training")
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="List of input JSONL files to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/train.jsonl",
        help="Output file path (default: datasets/train.jsonl)"
    )
    parser.add_argument(
        "--reasoning-ratio",
        type=float,
        default=0.75,
        help="Ratio of examples to include reasoning (default: 0.75 for 75%%)"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    input_files = [base_dir / source for source in args.sources]
    output_file = base_dir / args.output
    
    print("="*80)
    print("TypeScript Dataset Preprocessing for Qwen3")
    print("="*80)
    print(f"Input files: {len(input_files)}")
    print(f"Output file: {output_file}")
    print(f"Reasoning ratio: {args.reasoning_ratio * 100:.1f}%")
    print("="*80)
    
    stats = process_dataset(input_files, output_file, reasoning_ratio=args.reasoning_ratio)
    
    print("\n" + "="*80)
    print("Statistics")
    print("="*80)
    print(f"Total loaded: {stats['total_loaded']:,}")
    print(f"Converted: {stats['converted']:,}")
    print(f"With reasoning: {stats['with_reasoning']:,} ({stats['with_reasoning']/stats['converted']*100:.1f}%)" if stats['converted'] > 0 else "With reasoning: 0")
    print(f"Direct output: {stats['direct_output']:,} ({stats['direct_output']/stats['converted']*100:.1f}%)" if stats['converted'] > 0 else "Direct output: 0")
    print(f"Invalid: {stats['invalid']:,}")
    print(f"Skipped: {stats['skipped']:,}")
    print(f"Duplicates removed: {stats['duplicates_removed']:,}")
    print(f"Final count: {stats['final_count']:,}")
    print("="*80)
    
    print(f"\n[OK] Preprocessed dataset saved to {output_file}")
    print(f"     File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

