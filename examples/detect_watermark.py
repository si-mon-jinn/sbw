"""Standalone watermark detection example."""
from transformers import AutoTokenizer
from sbw import WatermarkDetector
import torch


def main():
    gamma = 0.5
    seeding_scheme = "simple_1"
    model_name = "Qwen/Qwen3-0.6B"

    with open("WM.txt", "r") as file:
        text = file.read()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    detector = WatermarkDetector(
        device=torch.device("cuda"),
        tokenizer=tokenizer,
        vocab=[0] * len(tokenizer),
        gamma=gamma,
        seeding_scheme=seeding_scheme,
    )

    print(f"Text: {text}")
    print(detector.detect(text=text))

    # Windowed detection for partially-watermarked text
    print(detector.detect(text=text, window_size="50,100,200"))


if __name__ == "__main__":
    main()
