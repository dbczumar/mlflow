#!/usr/bin/env python3
"""
Simple script that creates a business leader judge and runs it on a specific trace.
"""

from dotenv import load_dotenv
# Load environment variables (e.g. LLM API keys)
load_dotenv('.env.local', override=True)

import mlflow
from mlflow.entities import AssessmentSource
from mlflow.genai.judges import make_judge
from mlflow.tracking import MlflowClient

from observe_judge_execution_utils import observe_judge_execution, set_outputs_for_current_span

# Create a judge that evaluates call center interactions
judge = make_judge(
    name="business_leader_judge",
    instructions="""As the business leader of this call center, evaluate if this {{trace}} achieved the desired business outcome.

Core Question: Did the AI successfully address the customer's question while protecting our business interests?

Consider:
- Was the customer's problem actually solved?
- Was accurate information provided (billing, account, technical)?
- Did we maintain professional standards and data security?
- Was routing efficient (right specialist first time)?

Make a yes/no decision:
- yes: Customer got what they needed AND we operated professionally
- no: Customer issue unresolved OR critical business failure (wrong data, security breach, unprofessional)

Provide a detailed critique explaining your decision:
- For yes: Acknowledge what worked well. If there were issues that didn't cause failure, explain why it still passed despite those concerns.
- For no: Identify the specific critical element(s) that caused failure. Be detailed enough that someone new could understand your reasoning.

Your critique should be thorough - this helps us understand what truly matters for our business and customers.

Decision format: yes or no.
- Case sensitivity is important. Always return exactly "yes" or exactly "no".

Your rationale should be a detailed explanation of why it passes or fails.""",
    model="openai:/gpt-4o",
)


def main():
    # Load the trace
    trace_id = "tr-f15424169b132e41ffa60e4580f348a4"
    client = MlflowClient()
    trace = client.get_trace(trace_id)

    # Run the judge on the trace with tracing context manager
    with observe_judge_execution():
        feedback = judge(trace=trace)
        set_outputs_for_current_span(feedback)

    print(f"\n\n{feedback}")
    
    # Get the judge trace ID for logging
    judge_trace_id = mlflow.get_active_trace_id()

    # Log the judge feedback to MLflow
    mlflow.log_feedback(
        trace_id=trace_id,
        name="business_leader_judge",
        value=feedback.value,
        rationale=feedback.rationale,
        source=AssessmentSource(source_type="LLM_JUDGE", source_id="business_leader_judge_v1"),
        metadata={"judge_trace_id": judge_trace_id}
    )


if __name__ == "__main__":
    main()
