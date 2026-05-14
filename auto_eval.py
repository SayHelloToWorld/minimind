import os
import sys
import json
import time
import argparse
import warnings
import tempfile
import shutil
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import get_model_params

# Reuse init_model from eval_llm.py — same logic for loading native/huggingface weights
sys.path.insert(0, os.path.dirname(__file__))
from eval_llm import init_model

warnings.filterwarnings('ignore')


def save_as_transformers(model, tokenizer, output_dir):
    """
    Save a native MiniMind model in HuggingFace Transformers format
    so that lm_eval can load it.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Ensure config is proper for AutoModel loading
    config = model.config
    if hasattr(config, 'model_type') and config.model_type == 'minimind':
        # Copy the model source file so from_pretrained can import it
        src_model_file = os.path.join(os.path.dirname(__file__), 'model', 'model_minimind.py')
        if os.path.exists(src_model_file):
            shutil.copy2(src_model_file, os.path.join(output_dir, 'model_minimind.py'))
        # Also copy tokenizer files if needed
        src_tokenizer_dir = os.path.join(os.path.dirname(__file__), 'model')
        for fname in ['tokenizer.json', 'tokenizer_config.json']:
            src = os.path.join(src_tokenizer_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, fname))


# ============================================================
# lm_eval via CLI
# ============================================================
def check_lm_eval_installed():
    """Check if lm_eval CLI is available."""
    try:
        subprocess.run(['lm_eval', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_lm_eval_cli(model_path, tasks, device, batch_size, apply_chat_template,
                    num_fewshot=0, output_path=None):
    """
    Run benchmarks using the lm_eval CLI (not Python API).
    model_path: path to the HuggingFace model directory.
    """
    if not check_lm_eval_installed():
        print("[Error] lm_eval is not installed or not found in PATH.")
        print("  Install it via: pip install lm-eval")
        print("  Or: git clone https://github.com/EleutherAI/lm-evaluation-harness && cd lm-evaluation-harness && pip install -e .")
        sys.exit(1)

    # Build the lm_eval command
    model_args_str = f"pretrained={model_path},dtype=float16,device={device}"
    tasks_str = ','.join(tasks)

    cmd = [
        'lm_eval',
        '--model', 'hf',
        '--model_args', model_args_str,
        '--tasks', tasks_str,
        '--batch_size', str(batch_size),
        '--num_fewshot', str(num_fewshot),
    ]

    if apply_chat_template:
        cmd += ['--apply_chat_template']

    if output_path:
        cmd += ['--output_path', output_path]

    print(f"\n{'='*60}")
    print(f"Running lm_eval CLI on: {model_path}")
    print(f"Tasks: {tasks_str}")
    print(f"Device: {device} | batch_size: {batch_size}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run the command and stream output in real-time
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[Error] lm_eval failed with return code {result.returncode}")
        if result.stderr:
            print(f"stderr: {result.stderr}")
        sys.exit(1)

    if output_path:
        print(f"\n[Results saved to {output_path}]")

    return True


def parse_lm_eval_results(output_path, tasks):
    """
    Parse lm_eval results from the generated JSON files.
    lm_eval saves results to: {output_path}/{model_name}/results_{timestamp}.json
    """
    results_dir = output_path
    if not os.path.isdir(results_dir):
        # Look for subdirectories
        for entry in os.listdir(results_dir):
            full_path = os.path.join(results_dir, entry)
            if os.path.isdir(full_path):
                json_files = sorted(
                    [f for f in os.listdir(full_path) if f.startswith('results_') and f.endswith('.json')],
                    reverse=True
                )
                if json_files:
                    result_path = os.path.join(full_path, json_files[0])
                    break
        else:
            return {}
    else:
        json_files = sorted(
            [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.json')],
            reverse=True
        )
        if not json_files:
            return {}
        result_path = os.path.join(results_dir, json_files[0])

    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task_results = {}
    if 'results' in data:
        for task_name in tasks:
            if task_name in data['results']:
                task_data = data['results'][task_name]
                acc = task_data.get('acc,none', task_data.get('acc', None))
                acc_norm = task_data.get('acc_norm,none', task_data.get('acc_norm', None))
                # Also capture perplexity if available (word_perplexity, ppl, etc.)
                ppl = task_data.get('word_perplexity,none', task_data.get('word_perplexity', None))
                if ppl is None:
                    ppl = task_data.get('perplexity,none', task_data.get('perplexity', None))
                task_results[task_name] = {
                    'acc': acc,
                    'acc_norm': acc_norm,
                    'perplexity': ppl,
                }
    return task_results


def print_task_summary(task_results, title="Task Performance Summary"):
    """Print a formatted summary table of task results."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    # Check if any results have perplexity
    has_ppl = any(res.get('perplexity') is not None for res in task_results.values())
    if has_ppl:
        header = f"{'Task':<25} {'PPL':<12} {'Acc':<10} {'Acc_Norm':<10}"
        print(header)
        print("-" * 60)
        for task, res in task_results.items():
            ppl_str = f"{res['perplexity']:.4f}" if res.get('perplexity') is not None else 'N/A'
            acc_str = f"{res['acc']:.4f}" if res.get('acc') is not None else 'N/A'
            acc_norm_str = f"{res['acc_norm']:.4f}" if res.get('acc_norm') is not None else 'N/A'
            print(f"{task:<25} {ppl_str:<12} {acc_str:<10} {acc_norm_str:<10}")
    else:
        header = f"{'Task':<25} {'Acc':<10} {'Acc_Norm':<10}"
        print(header)
        print("-" * 50)
        for task, res in task_results.items():
            acc_str = f"{res['acc']:.4f}" if res.get('acc') is not None else 'N/A'
            acc_norm_str = f"{res['acc_norm']:.4f}" if res.get('acc_norm') is not None else 'N/A'
            print(f"{task:<25} {acc_str:<10} {acc_norm_str:<10}")
    print("=" * 60)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MiniMind 模型自动评估 (via lm_eval CLI)")
    # Model args (mirrors eval_llm.py)
    parser.add_argument('--load_from', default='model', type=str,
                        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='checkpoints', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str,
                        help="权重名称前缀（pretrain, full_sft, rlhf, ppo_actor, grpo, 等）")
    parser.add_argument('--lora_weight', default='None', type=str,
                        help="LoRA权重名称（None表示不使用）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--inference_rope_scaling', default=True, action='store_true',
                        help="启用RoPE位置编码外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        type=str, help="运行设备")

    # Evaluation args
    parser.add_argument('--eval_ppl', action='store_true', default=False,
                        help="评估模型困惑度（PPL，通过lm_eval的wikitext等任务）")
    parser.add_argument('--eval_benchmark', action='store_true', default=False,
                        help="评估标准benchmark（需安装lm_eval）")
    parser.add_argument('--tasks', nargs='+', type=str,
                        default=['ceval-valid', 'cmmlu', 'arc_easy', 'piqa',
                                 'openbookqa', 'hellaswag', 'social_iqa'],
                        help="传给--eval_benchmark的任务列表")
    parser.add_argument('--ppl_tasks', nargs='+', type=str,
                        default=['wikitext'],
                        help="传给--eval_ppl的任务列表（如wikitext, lambada_standard等）")
    parser.add_argument('--num_fewshot', default=0, type=int, help="few-shot数量")
    parser.add_argument('--batch_size', default=16, type=int, help="评估batch_size")
    parser.add_argument('--apply_chat_template', action='store_true', default=False,
                        help="是否使用chat_template（指令模型需要）")
    parser.add_argument('--output_dir', default='eval_results', type=str,
                        help="评估结果输出目录")
    parser.add_argument('--temp_dir', default=None, type=str,
                        help="临时HF格式模型保存目录（默认使用系统临时目录）")
    parser.add_argument('--keep_temp', action='store_true', default=False,
                        help="保留临时HF格式模型目录（便于调试）")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Initialize model and tokenizer
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Initializing model...")
    print(f"  load_from = {args.load_from}")
    print(f"  weight    = {args.weight}")
    print(f"  hidden    = {args.hidden_size}")
    print(f"  moe       = {bool(args.use_moe)}")
    print(f"  device    = {args.device}")
    print(f"{'='*60}\n")

    model, tokenizer = init_model(args)

    # ----------------------------------------------------------
    # Helper: convert native model to HF format for lm_eval
    # ----------------------------------------------------------
    def prepare_model_path():
        is_native = ('model' in args.load_from)
        if not is_native:
            return args.load_from, None

        temp_dir_owner = None
        model_path = args.temp_dir
        if not model_path:
            temp_dir_owner = tempfile.TemporaryDirectory(prefix='minimind_eval_')
            model_path = temp_dir_owner.name

        print(f"\nSaving native model to Transformers format at: {model_path}")
        save_as_transformers(model, tokenizer, model_path)

        # Copy model source for custom model_type resolution
        src_model_file = os.path.join(os.path.dirname(__file__), 'model', 'model_minimind.py')
        if os.path.exists(src_model_file):
            dst = os.path.join(model_path, 'model_minimind.py')
            if not os.path.exists(dst):
                shutil.copy2(src_model_file, dst)

        return model_path, temp_dir_owner

    # ----------------------------------------------------------
    # Perplexity Evaluation (via lm_eval CLI)
    # ----------------------------------------------------------
    if args.eval_ppl:
        print("\n" + "-" * 50)
        print("Evaluating Perplexity (PPL) via lm_eval...")
        print("-" * 50)

        model_path, temp_dir_owner = prepare_model_path()
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.output_dir, f'ppl_{timestamp}')

        run_lm_eval_cli(
            model_path=model_path,
            tasks=args.ppl_tasks,
            device=args.device,
            batch_size=args.batch_size,
            apply_chat_template=False,  # PPL tasks don't use chat template
            num_fewshot=0,  # PPL tasks are zero-shot
            output_path=output_path,
        )

        # Parse and display PPL results
        task_results = parse_lm_eval_results(output_path, args.ppl_tasks)
        print_task_summary(task_results, "Perplexity Summary")

        # Clean up temp directory
        if temp_dir_owner is not None and not args.keep_temp:
            temp_dir_owner.cleanup()
            print(f"\n[Temporary model directory cleaned up]")
        elif temp_dir_owner is not None:
            print(f"\n[Temporary model directory kept at: {model_path}]")

    # ----------------------------------------------------------
    # Benchmark Evaluation (via lm_eval CLI)
    # ----------------------------------------------------------
    if args.eval_benchmark:
        print("\n" + "-" * 50)
        print("Evaluating Benchmark via lm_eval...")
        print("-" * 50)

        model_path, temp_dir_owner = prepare_model_path()
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.output_dir, f'benchmark_{timestamp}')

        run_lm_eval_cli(
            model_path=model_path,
            tasks=args.tasks,
            device=args.device,
            batch_size=args.batch_size,
            apply_chat_template=args.apply_chat_template,
            num_fewshot=args.num_fewshot,
            output_path=output_path,
        )

        # Parse and display results
        task_results = parse_lm_eval_results(output_path, args.tasks)
        print_task_summary(task_results, "Task Performance Summary")

        # Clean up temp directory
        if temp_dir_owner is not None and not args.keep_temp:
            temp_dir_owner.cleanup()
            print(f"\n[Temporary model directory cleaned up]")
        elif temp_dir_owner is not None:
            print(f"\n[Temporary model directory kept at: {model_path}]")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()