"""
Constants for the InstructionsJudge module.

This module contains constant values used by the InstructionsJudge class,
including the augmented prompt template for trace-based evaluation.
"""

# Augmented prompt template for trace-based evaluation
INSTRUCTIONS_JUDGE_TRACE_PROMPT_TEMPLATE = """
You have access to tools to analyze the trace. You MUST follow this methodology:

EFFICIENCY AND THOROUGHNESS PRINCIPLE:
Unless otherwise specified in the instructions, you should be as FAST/EFFICIENT *and* THOROUGH
as possible. If the problem is challenging or complex, prefer THOROUGHNESS over efficiency -
it's better to be comprehensive than to miss important details.

REQUIRED STEPS (call these three tools in parallel for efficiency):
1. ALWAYS fetch the trace metadata to understand the overall context, timing, and execution details
2. ALWAYS retrieve the root span to understand the top-level inputs and outputs of the interaction.
   The root span typically contains the overall inputs to the agent and the final outputs.
3. ALWAYS list all spans to see the complete trace structure and understand the flow of execution

IMPORTANT: Make all three required tool calls IN A SINGLE MESSAGE for best performance.
The tools can execute in parallel.

After completing these required steps, use more tools *if and only if* needed. For example:
- Retrieve specific spans by ID to examine their details
- Search for patterns or specific text across the entire trace
- Continue using tools until you have gathered sufficient information

Remember: Be efficient where possible, but thorough when necessary. Quality of evaluation
takes precedence over speed when dealing with complex traces.
"""
