"""Offline vLLM generation with watermarking example."""
import torch
from vllm import LLM, SamplingParams
from vllm_sbw import SBWLogitsProcessor
from sbw import WatermarkDetector


def main():
    gamma = 0.5
    delta = 2.0
    seeding_scheme = "simple_1"

    model_name = "Qwen/Qwen3-0.6B"
    llm = LLM(
        model=model_name,
        max_model_len=6000,
        max_num_batched_tokens=6000,
        gpu_memory_utilization=0.8,
        max_num_seqs=10,
        seed=1234,
        logits_processors=[SBWLogitsProcessor]
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
        extra_args={"gamma": gamma, "delta": delta, "seeding_scheme": seeding_scheme}
    )

    output = llm.generate(prompts=["Hello! How are you?"], sampling_params=sampling_params)
    text = output[0].outputs[0].text

    detector = WatermarkDetector(
        device=torch.device("cuda"),
        tokenizer=llm.llm_engine.tokenizer,
        vocab=[0] * llm.llm_engine.model_config.get_vocab_size(),
        gamma=gamma,
        seeding_scheme=seeding_scheme,
    )

    print(f"Generated text: {text}")
    print(detector.detect(text=text))


if __name__ == "__main__":
    main()
