# Qwen3 Colab fine-tuning starter

This folder adds a reproducible **LoRA/QLoRA supervised fine-tuning** starter for the smallest original Qwen3 dense checkpoint, **Qwen/Qwen3-0.6B**.

> This trains an adapter; it does not erase safety alignment embedded in the base model. The sample data is only a smoke-test dataset. Replace it with a reviewed dataset for meaningful results. The examples are designed to improve helpfulness on benign requests while retaining appropriate safety boundaries.

## Run from an Android phone

1. Open the notebook in Google Colab: [Open in Colab](https://colab.research.google.com/github/mdarman0964/Qwen3/blob/feature/colab-finetuning-starter/training/Qwen3_0.6B_Colab_LoRA.ipynb).
2. In Colab choose **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the cells from top to bottom.
4. The notebook clones this branch and runs `training/train.py`.
5. The adapter is saved under `/content/qwen3-0.6b-lora` and downloaded as a ZIP by the final cell.

## Dataset format

Create UTF-8 JSONL files with one JSON object per line:

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"How do I protect my account?"},{"role":"assistant","content":"Use a unique password, enable MFA, and keep recovery details current."}]}
```

Put training examples in `training/data/train.jsonl`. See `training/data/sample.jsonl` for examples. Do not put passwords, API keys, personal records, or other sensitive data in a training set.

## Local command

```bash
pip install -r training/requirements.txt
python training/train.py --data_path training/data/sample.jsonl --output_dir ./qwen3-0.6b-lora --max_steps 20
```

For a real run, use a larger, high-quality dataset and remove `--max_steps 20` (or set a suitable value). A small free Colab GPU may run out of memory; reduce `--max_length` to 512 and `--batch_size` to 1.

## Safety and evaluation

- Keep a separate held-out evaluation set; do not evaluate only on training examples.
- Include ordinary benign prompts that previously triggered false refusals, along with examples that should still be refused or redirected.
- Review all training examples before use. Do not train the model to provide instructions for wrongdoing, abuse, or evading safeguards.
- A LoRA adapter changes model behavior but does not guarantee policy compliance. Add and test explicit input/output moderation in the application if required.

## Base model

[Qwen/Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B). Review the model card and license before distribution.
