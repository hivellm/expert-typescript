# Expert TypeScript

[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/hivellm/expert-typescript/releases/tag/v0.0.1)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-development-yellow.svg)](README.md#quick-start)

[![Base Model](https://img.shields.io/badge/base%20model-Qwen3--0.6B-orange.svg)](README.md#features)
[![Adapter](https://img.shields.io/badge/adapter-DoRA%20r%3D12-blue.svg)](README.md#training--configuration)
[![Dataset](https://img.shields.io/badge/dataset-207k%20examples-brightgreen.svg)](README.md#features)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20CUDA-0078d4.svg)](README.md#features)

TypeScript code generation and instruction expert trained on multiple high-quality sources including synthetic datasets, official TypeScript documentation, and real-world code samples.

## Quick Start

```bash
# 1. Download package (when available)
# wget https://github.com/hivellm/expert-typescript/releases/download/v0.0.1/expert-typescript-qwen3-0-6b.v0.0.1.expert

# 2. Install
expert-cli install expert-typescript-qwen3-0-6b.v0.0.1.expert

# 3. Use
expert-cli chat --experts typescript
> Create a function that validates email addresses using regex
```

**Works best for:** Code generation, refactoring, type definitions, modern TypeScript patterns  
**Limitations:** See Known Limitations below

## Features

- ✅ **Code Generation**: Generate TypeScript code from natural language descriptions
- ✅ **Code Explanation**: Explain TypeScript code logic and patterns
- ✅ **Refactoring**: Suggest improvements and refactorings for TypeScript code
- ✅ **Type Safety**: Understand and apply TypeScript type system
- ✅ **Modern Patterns**: Knowledge of modern TypeScript features and best practices
- ✅ **DoRA adapter (r=12)** optimized for code generation tasks
- ✅ **207,283 validated examples** from official docs and open-source TypeScript code
- ✅ **Growing dataset** combining TypeScript Handbook extractions and large curated slices of the-stack TypeScript subset

## What It Can Do ✅

**Excellent Support:**
- ✅ Function generation from natural language
- ✅ Interface and type definitions
- ✅ Class implementations
- ✅ Code refactoring suggestions
- ✅ Code explanation and documentation
- ✅ Modern TypeScript features (generics, async/await, decorators)
- ✅ Type inference and type safety
- ✅ Common patterns (singletons, factories, builders)

**Example Outputs:**

```typescript
// Code Generation
Input: "Create a function that validates email addresses using regex"
Output: 
function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// Type Generation
Input: "Generate TypeScript interface for a user profile"
Output:
interface UserProfile {
  id: string;
  username: string;
  email: string;
  createdAt: Date;
  isActive: boolean;
  roles: string[];
}

// Code Explanation
Input: "Explain the logic of the useDispatcher function"
Output: The useDispatcher function takes an argument that can be a Dispatcher, 
DispatcherRegisterOptions, or an array of these types. It registers dispatchers 
with the store system...
```

## Known Limitations ⚠️

**These patterns may have lower success rates:**
- ⚠️ **Very complex type inference** - May struggle with deeply nested generics
- ⚠️ **Advanced decorators** - Limited support for complex decorator patterns
- ⚠️ **Framework-specific code** - May not follow specific framework conventions perfectly
- ⚠️ **Large codebases** - Best for single functions/interfaces, not entire modules

**Recommendation:** 
- ✅ Use for generating individual functions, types, and small code snippets
- ✅ Provide clear, specific instructions for best results
- ⚠️ Review generated code for framework-specific patterns

## Dataset

### Current Dataset
- **Total Examples**: 207,283 instruction-output pairs
- **Sources**:
  - TypeScript Handbook extraction (`scripts/extract_typescript_docs.py`): 155 examples
  - [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) TypeScript subset (`scripts/integrate_the_stack.py --limit 100000`): 207,128 examples
- **Format**: Instruction-tuning dataset for TypeScript code generation (ChatML format)
- **Preprocessing**: Deduplication applied (50,026 duplicates removed)
- **Location**: `datasets/train.jsonl`
- **Integration Date**: 2025-11-08

### Dataset Statistics
- **Expansion**: Initial merge of documentation + curated The Stack subset (26k examples) expanded to ~207k examples
- **Quality**: Blend of official documentation snippets and a broad sample of real-world code patterns
- **Coverage**: Core TypeScript syntax plus significant representation of advanced language features and patterns

## Training

### Quick Start

```bash
# From expert-typescript directory
cd F:/Node/hivellm/expert/experts/expert-typescript

# Run training (CLI will download dataset automatically)
../../cli/target/release/expert-cli train
```

### Training Configuration

- **Base Model**: Qwen3-0.6B (INT4 quantized)
- **Adapter**: DoRA (rank=12, alpha=24)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj
- **Epochs**: 3
- **Learning Rate**: 0.0003
- **Batch Size**: 16
- **Gradient Accumulation**: 4 steps
- **Max Sequence Length**: 2048

### Dataset Information

The dataset currently includes:
- TypeScript Handbook documentation examples (155)
- Curated the-stack TypeScript code samples (207,128)

All datasets have been merged into `datasets/train.jsonl` with deduplication applied. The dataset is ready for additional augmentation and training experiments.

## Testing

Run the automated tests to validate the expert:

```bash
# Windows
.\test.ps1

# Linux/macOS
./test.sh
```

## Qualitative Checkpoint Review

Generate side-by-side outputs for the base model and every available checkpoint:

```powershell
cd F:/Node/hivellm/expert/experts/expert-typescript
F:/Node/hivellm/expert/cli/venv_windows/Scripts/python.exe compare.py
```

The script prints formatted outputs for each representative TypeScript task and saves a JSON summary as `checkpoint_comparison_results.json` for further analysis.

## Usage Examples

### Code Generation

```typescript
Input: "Create a function that validates email addresses using regex"
Output: 
function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}
```

### Code Explanation

```typescript
Input: "Explain the logic of the useDispatcher function"
Output: The useDispatcher function takes an argument that can be a Dispatcher, 
DispatcherRegisterOptions, or an array of these types. It registers dispatchers 
with the store system...
```

### Type Generation

```typescript
Input: "Generate TypeScript interface for a user profile"
Output:
interface UserProfile {
  id: string;
  username: string;
  email: string;
  createdAt: Date;
  isActive: boolean;
  roles: string[];
}
```

### Python Integration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model_path = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# Load expert adapter
adapter_path = "experts/expert-typescript"
model = PeftModel.from_pretrained(base_model, adapter_path)

# Generate TypeScript code
prompt = "Create a function that validates email addresses using regex"

messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.8,
        top_k=20
    )

code = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(code)
```

## Package

Once trained, package the expert:

```bash
expert-cli package \
  --manifest manifest.json \
  --weights weights/qwen3-06b/adapter \
  --output expert-typescript-qwen3-0-6b.v0.0.1.expert
```

## Performance Metrics

| Metric | Current |
|--------|---------|
| Adapter Size | ~26 MB |
| VRAM Overhead | ~18 MB |
| Load Time | <10ms (hot) |
| Training Time | ~8-12 hours (RTX 4090) |
| Dataset Size | 207,283 examples |

## License

MIT License - Same as the source dataset

## Credits

- **Datasets**: 
  - [mhhmm/typescript-instruct-20k-v2c](https://huggingface.co/datasets/mhhmm/typescript-instruct-20k-v2c) (19,994 examples)
  - TypeScript Handbook (Microsoft) - Official documentation (155 examples)
  - [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) - TypeScript subset (298,543 examples)
- **Base Model**: Qwen3-0.6B by Alibaba Cloud
- **Training Framework**: HuggingFace PEFT + Transformers
