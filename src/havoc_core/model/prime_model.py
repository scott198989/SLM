"""
HAVOC-7B PRIME Model

Integrates PRIME meta-reasoning directly into the model's generation process.
This is the main model class that users interact with.
"""

from __future__ import annotations

import json
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from havoc_core.config_7b import Havoc7BConfig, ReasoningTokenConfig
from havoc_core.model.transformer import HavocModel
from havoc_core.reasoning_tokens import (
    ReasoningTokenFormatter,
    ReasoningTokenParser,
    validate_reasoning_tokens
)
from havoc_core.tool_interface import ToolRegistry, ToolResult

# PRIME components (already exist in src/havoc_prime/)
from havoc_prime.router import TaskRouter, Budget, RoutingDecision
from havoc_prime.operator_graph import OperatorGraphBuilder, OperatorGraph
from havoc_prime.workspace import GlobalWorkspace
from havoc_prime.adversarial import Advocate, HavocAttack, Pragmatist, AdversarialSynthesizer
from havoc_prime.verification import GlobalVerification
from havoc_prime.compression import FinalCompression


@dataclass
class GenerationResult:
    """Result from PRIME generation"""
    generated_ids: torch.Tensor  # Generated token IDs
    text: str  # Decoded text (includes reasoning tokens)
    reasoning_segments: List[Any]  # Parsed reasoning segments
    tool_results: List[ToolResult]  # Results from tool calls
    workspace: GlobalWorkspace  # Final workspace state
    verification: Dict[str, Any]  # Verification report
    routing: RoutingDecision  # Routing decision
    metadata: Dict[str, Any]  # Additional metadata


class HavocPrimeModel(nn.Module):
    """
    HAVOC-7B with integrated PRIME meta-reasoning

    This model combines:
    - Base 7B transformer
    - PRIME meta-reasoning (budget-based, adversarial, etc.)
    - Reasoning token system (visible chain-of-thought)
    - Tool-calling interface

    Usage:
        model = HavocPrimeModel.from_pretrained("checkpoints/havoc_7b")
        result = model.generate_with_prime(
            prompt="Design a Box-Behnken DOE",
            tools=ToolRegistry(),
            max_new_tokens=512
        )
    """

    def __init__(
        self,
        config: Havoc7BConfig,
        tokenizer: Any = None
    ):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer

        # Base transformer model
        self.base_model = HavocModel(config)

        # Reasoning token system
        self.reasoning_config = ReasoningTokenConfig()
        self.reasoning_formatter = ReasoningTokenFormatter()
        self.reasoning_parser = ReasoningTokenParser()

        # Tool registry
        self.tool_registry = ToolRegistry()

        # PRIME components
        self.router = TaskRouter()
        self.graph_builder = OperatorGraphBuilder()
        self.advocate = Advocate()
        self.attack = HavocAttack()
        self.pragmatist = Pragmatist()
        self.synthesizer = AdversarialSynthesizer()
        self.verification = GlobalVerification()
        self.compression = FinalCompression()

    def forward(self, *args, **kwargs):
        """Forward pass - delegate to base model"""
        return self.base_model(*args, **kwargs)

    def generate_with_prime(
        self,
        prompt: str,
        tokenizer: Any = None,
        tools: Optional[ToolRegistry] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        enable_prime: bool = True,
        force_budget: Optional[Budget] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate with PRIME meta-reasoning

        Args:
            prompt: Input prompt
            tokenizer: Tokenizer (uses self.tokenizer if None)
            tools: Tool registry (uses self.tool_registry if None)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            enable_prime: Enable PRIME reasoning (if False, direct generation)
            force_budget: Force specific budget (for testing)
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult object
        """
        tokenizer = tokenizer or self.tokenizer
        tools = tools or self.tool_registry

        if tokenizer is None:
            raise ValueError("No tokenizer provided. Pass tokenizer argument or set self.tokenizer")

        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.base_model.embed_tokens.weight.device)

        # Step 1: Route task to determine budget
        routing = self.router.route(prompt)

        if force_budget:
            routing.budget = force_budget

        # Step 2: Handle MICRO budget or disabled PRIME (direct generation)
        if not enable_prime or routing.budget == Budget.MICRO:
            return self._direct_generate(
                prompt, prompt_ids, tokenizer, max_new_tokens, temperature, top_p, top_k, routing
            )

        # Step 3: Build operator graph
        operator_graph = self.graph_builder.build_graph(
            prompt, routing.task_type, routing.budget
        )

        # Step 4: Generate with PRIME reasoning
        return self._generate_with_reasoning(
            prompt, prompt_ids, tokenizer, operator_graph, routing, tools,
            max_new_tokens, temperature, top_p, top_k
        )

    def _direct_generate(
        self,
        prompt: str,
        prompt_ids: torch.Tensor,
        tokenizer: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        routing: RoutingDecision
    ) -> GenerationResult:
        """Direct generation without PRIME reasoning"""

        # Simple generation using base model
        generated_ids = self.base_model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            tokenizer=tokenizer
        )

        # Decode
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        # Create minimal result
        workspace = GlobalWorkspace()

        return GenerationResult(
            generated_ids=generated_ids,
            text=text,
            reasoning_segments=[],
            tool_results=[],
            workspace=workspace,
            verification={"passed": True, "issues": [], "warnings": []},
            routing=routing,
            metadata={"budget": routing.budget.value, "prime_enabled": False}
        )

    def _generate_with_reasoning(
        self,
        prompt: str,
        prompt_ids: torch.Tensor,
        tokenizer: Any,
        operator_graph: OperatorGraph,
        routing: RoutingDecision,
        tools: ToolRegistry,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> GenerationResult:
        """
        Generate with full PRIME reasoning

        This implements the PRIME reasoning loop:
        1. For each subgoal in operator graph:
           a. Generate <reason>...</reason> (chain-of-thought)
           b. If tool needed, generate <tool>...</tool> and execute
           c. If adversarial enabled, generate <advocate>, <attack>, <pragmatist>
           d. Update workspace
        2. Global verification
        3. Final answer generation
        """

        device = prompt_ids.device
        workspace = GlobalWorkspace()
        tool_results = []
        reasoning_segments = []

        # Initialize generation state
        current_ids = prompt_ids
        generated_tokens = []
        max_tokens_remaining = max_new_tokens

        # Process each subgoal
        for subgoal in operator_graph.subgoals:
            # Check if we have tokens left
            if max_tokens_remaining <= 0:
                break

            # Step A: Generate reasoning for this subgoal
            reason_prompt = f"\n<reason>\nSubgoal: {subgoal.description}\n"
            reason_ids = tokenizer.encode(reason_prompt, return_tensors="pt").to(device)

            # Generate reasoning content
            reason_generated = self._generate_segment(
                torch.cat([current_ids, reason_ids], dim=1),
                tokenizer,
                max_new_tokens=min(200, max_tokens_remaining),
                temperature=temperature,
                stop_tokens=[self.reasoning_config.get_token_id("</reason>")]
            )

            # Add closing tag
            reason_close_ids = tokenizer.encode("\n</reason>", return_tensors="pt").to(device)
            current_ids = torch.cat([current_ids, reason_ids, reason_generated, reason_close_ids], dim=1)
            max_tokens_remaining -= (reason_generated.shape[1] + reason_ids.shape[1] + reason_close_ids.shape[1])

            # Step B: Check if tool is needed for this subgoal
            if subgoal.required_tools:
                for tool_name in subgoal.required_tools:
                    # Generate tool call
                    tool_prompt = f"\n<tool>\n"
                    tool_ids = tokenizer.encode(tool_prompt, return_tensors="pt").to(device)

                    # Generate tool call JSON
                    tool_call_generated = self._generate_segment(
                        torch.cat([current_ids, tool_ids], dim=1),
                        tokenizer,
                        max_new_tokens=min(150, max_tokens_remaining),
                        temperature=0.3,  # Lower temperature for structured JSON
                        stop_tokens=[self.reasoning_config.get_token_id("</tool>")]
                    )

                    tool_close_ids = tokenizer.encode("\n</tool>", return_tensors="pt").to(device)
                    current_ids = torch.cat([current_ids, tool_ids, tool_call_generated, tool_close_ids], dim=1)
                    max_tokens_remaining -= (tool_call_generated.shape[1] + tool_ids.shape[1] + tool_close_ids.shape[1])

                    # Execute tool
                    tool_call_text = tokenizer.decode(tool_call_generated[0], skip_special_tokens=True)
                    try:
                        tool_result = tools.execute_from_json(tool_call_text)
                        tool_results.append(tool_result)

                        # Add tool result to context
                        tool_result_text = f"\n\nTool result: {tool_result.to_json()}\n"
                        tool_result_ids = tokenizer.encode(tool_result_text, return_tensors="pt").to(device)
                        current_ids = torch.cat([current_ids, tool_result_ids], dim=1)
                        max_tokens_remaining -= tool_result_ids.shape[1]

                        # Store in workspace
                        workspace.store_partial_result(subgoal.id, tool_result.to_dict())

                    except Exception as e:
                        print(f"Tool execution failed: {str(e)}")

            # Step C: Adversarial reasoning (if budget allows)
            if routing.budget in {Budget.MEDIUM, Budget.HEAVY}:
                # Generate advocate argument
                advocate_text = self._generate_adversarial_segment(
                    current_ids, tokenizer, "advocate", max_tokens_remaining, temperature
                )
                current_ids, tokens_used = advocate_text
                max_tokens_remaining -= tokens_used

                # Generate attack argument
                attack_text = self._generate_adversarial_segment(
                    current_ids, tokenizer, "attack", max_tokens_remaining, temperature
                )
                current_ids, tokens_used = attack_text
                max_tokens_remaining -= tokens_used

                # Generate pragmatist synthesis
                pragmatist_text = self._generate_adversarial_segment(
                    current_ids, tokenizer, "pragmatist", max_tokens_remaining, temperature
                )
                current_ids, tokens_used = pragmatist_text
                max_tokens_remaining -= tokens_used

        # Step D: Global verification
        verification_report = self.verification.verify(workspace, operator_graph)

        # Step E: Final answer generation
        final_prompt = "\n\nFinal answer: "
        final_ids = tokenizer.encode(final_prompt, return_tensors="pt").to(device)
        current_ids = torch.cat([current_ids, final_ids], dim=1)

        # Generate final answer
        final_generated = self._generate_segment(
            current_ids,
            tokenizer,
            max_new_tokens=max_tokens_remaining,
            temperature=temperature,
            stop_tokens=[tokenizer.eos_token_id]
        )

        current_ids = torch.cat([current_ids, final_generated], dim=1)

        # Decode full generation
        full_text = tokenizer.decode(current_ids[0], skip_special_tokens=False)

        # Parse reasoning segments
        reasoning_segments = self.reasoning_parser.parse(full_text)

        # Validate reasoning tokens
        is_valid, errors = validate_reasoning_tokens(full_text)
        if not is_valid:
            print(f"Warning: Reasoning token validation failed: {errors}")

        return GenerationResult(
            generated_ids=current_ids,
            text=full_text,
            reasoning_segments=reasoning_segments,
            tool_results=tool_results,
            workspace=workspace,
            verification={
                "passed": verification_report.passed,
                "issues": verification_report.issues,
                "warnings": verification_report.warnings
            },
            routing=routing,
            metadata={
                "budget": routing.budget.value,
                "prime_enabled": True,
                "num_subgoals": len(operator_graph.subgoals),
                "num_tool_calls": len(tool_results),
                "validation_passed": is_valid
            }
        )

    def _generate_segment(
        self,
        prompt_ids: torch.Tensor,
        tokenizer: Any,
        max_new_tokens: int,
        temperature: float,
        stop_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Generate a segment of text with optional stop tokens"""

        self.eval()
        generated = []
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                input_ids = prompt_ids if past_key_values is None else prompt_ids[:, -1:]
                logits, past_key_values = self.base_model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                # Sample next token
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check stop conditions
                if next_token.item() == tokenizer.eos_token_id:
                    break

                if stop_tokens and next_token.item() in stop_tokens:
                    break

                generated.append(next_token)
                prompt_ids = torch.cat([prompt_ids, next_token], dim=1)

        if generated:
            return torch.cat(generated, dim=1)
        else:
            return torch.tensor([[]], dtype=torch.long, device=prompt_ids.device)

    def _generate_adversarial_segment(
        self,
        current_ids: torch.Tensor,
        tokenizer: Any,
        segment_type: str,  # "advocate", "attack", or "pragmatist"
        max_tokens: int,
        temperature: float
    ) -> Tuple[torch.Tensor, int]:
        """Generate an adversarial reasoning segment"""

        device = current_ids.device

        # Start tag
        start_tag = f"\n<{segment_type}>\n"
        start_ids = tokenizer.encode(start_tag, return_tensors="pt").to(device)

        # Generate content
        stop_token_id = self.reasoning_config.get_token_id(f"</{segment_type}>")
        generated = self._generate_segment(
            torch.cat([current_ids, start_ids], dim=1),
            tokenizer,
            max_new_tokens=min(200, max_tokens),
            temperature=temperature,
            stop_tokens=[stop_token_id]
        )

        # End tag
        end_tag = f"\n</{segment_type}>"
        end_ids = tokenizer.encode(end_tag, return_tensors="pt").to(device)

        # Concatenate
        result_ids = torch.cat([current_ids, start_ids, generated, end_ids], dim=1)
        tokens_used = start_ids.shape[1] + generated.shape[1] + end_ids.shape[1]

        return result_ids, tokens_used

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        **kwargs
    ) -> "HavocPrimeModel":
        """
        Load model from checkpoint

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to load model on
            **kwargs: Additional arguments

        Returns:
            Loaded HavocPrimeModel
        """
        import os
        from pathlib import Path

        checkpoint_dir = Path(checkpoint_path)

        # Load config
        config = Havoc7BConfig.from_pretrained(checkpoint_path)

        # Create model
        model = cls(config)

        # Load weights
        model_file = checkpoint_dir / "model.pt"
        if model_file.exists():
            state_dict = torch.load(model_file, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded model weights from {model_file}")
        else:
            print(f"Warning: No model weights found at {model_file}")

        model.to(device)
        model.eval()

        return model

    def save_pretrained(self, save_path: str):
        """
        Save model to checkpoint

        Args:
            save_path: Path to save checkpoint
        """
        from pathlib import Path

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_path)

        # Save model weights
        model_file = save_dir / "model.pt"
        torch.save(self.state_dict(), model_file)
        print(f"Saved model to {model_file}")

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_num_params_billions(self) -> float:
        """Get number of parameters in billions"""
        return self.get_num_params() / 1e9

    def __repr__(self) -> str:
        params_b = self.get_num_params_billions()
        return (
            f"HavocPrimeModel(\n"
            f"  parameters={params_b:.2f}B,\n"
            f"  config={self.config.__class__.__name__},\n"
            f"  prime_enabled={self.config.enable_prime},\n"
            f"  tools={len(self.tool_registry.list_tools())}\n"
            f")"
        )
