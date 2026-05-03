## Project Overview

This project is an AI-powered loan intelligence system that combines deterministic financial computation with a conversational agent to provide real-time, context-aware financial guidance.

The system models user repayment behavior over time and evaluates how deviations from an expected repayment trajectory impact loan health. Based on this analysis, it classifies users into risk categories (LOW, MEDIUM, HIGH) and adapts its responses accordingly.

At its core, the system integrates a tool-augmented LLM that relies on structured financial computations rather than free-form reasoning. This ensures that all outputs related to EMI, interest, and outstanding balance remain accurate and grounded.

Additionally, the assistant incorporates sentiment-aware tone adaptation, allowing it to respond differently based on both user emotional state and repayment risk. This enables behavior-aware interaction that is closer to real-world financial advisory systems.

The project demonstrates:
- tool-grounded LLM reasoning for financial tasks
- real-time stateful decision making
- behavioral risk modeling based on repayment patterns
- adaptive response generation based on sentiment and risk

## Key Features

### 1. Tool-Augmented LLM Reasoning
- All financial responses are generated using tool calls
- Prevents hallucination by grounding outputs in deterministic computations
- Ensures correctness for EMI, interest, and balance-related queries

### 2. Behavioral Risk Modeling
- Computes expected principal using amortization logic
- Tracks actual principal based on user repayment behavior
- Calculates deviation and classifies users into LOW, MEDIUM, HIGH risk
- Enables behavior-aware decision making

### 3. Stateful Loan Tracking
- Maintains evolving loan state using session-based storage
- Tracks current month, principal, and payment activity
- Ensures consistent progression of loan lifecycle

### 4. Conversational AI Agent
- LLM-powered interface for user interaction
- Supports contextual conversations using short-term memory
- Integrates tightly with computation layer for grounded responses

### 5. Guardrail-Based Input Validation
- Filters unsafe or irrelevant inputs before processing
- Ensures system only responds to valid financial queries
- Acts as a control layer before LLM execution

### 6. Sentiment-Aware Processing
- Detects user sentiment (calm, neutral, agitated)
- Feeds emotional context into response generation

### 7. Adaptive Tone Control
- Combines sentiment and risk level to determine response tone
- Adjusts between supportive, corrective, and firm messaging
- Mimics escalation behavior seen in real financial systems

### 8. Deterministic Financial Computation Layer
- Handles EMI calculation, interest accrual, and principal updates
- Ensures consistency between backend logic and AI responses
- Serves as the single source of truth for all financial data

### 9. Robust State and Edge Handling
- Prevents duplicate payments within the same cycle
- Handles overpayment and final settlement conditions
- Applies interest correctly in absence of payment
- Locks system after loan closure

## Workflow


<img width="1072" height="691" alt="image" src="https://github.com/user-attachments/assets/0812c7eb-6763-4085-ae18-3c9b75ae5847" />

## Core Modules

### 1 Guardrail Layer (Input Validation)

The system uses a dedicated LLM-based guardrail to validate user inputs before any processing.

- Model: `openai/gpt-oss-safeguard-20b` (via Groq API)
- Output: binary decision (1 = allow, 0 = block)
- Temperature: 0 (deterministic behavior)

The guardrail focuses on **intent classification**, not keyword filtering. It allows:
- normal conversation (greetings, vague inputs)
- financial queries (EMI, payments, dues)

It blocks:
- prompt injection attempts (e.g., "ignore instructions")
- internal system probing (prompts, tools, memory)
- fraudulent or manipulative intent
- clearly unrelated task requests (e.g., coding tasks)

This ensures that only safe and relevant inputs reach the main agent.

---

### 2 Sentiment Analysis Module

User sentiment is extracted using an LLM-based structured output system.

- Model: `llama-3.3-70b-versatile` (via Groq API)
- Output format: strict JSON with one of:
  - `calm`
  - `neutral`
  - `agitated`

The system:
- analyzes tone, phrasing, and financial context
- detects subtle emotional signals (stress, frustration, compliance)
- avoids keyword-only classification

Implementation uses a **structured response schema (Pydantic)** to enforce correctness and consistency.

This sentiment is later used for **tone-aware response generation**.

---

### 3 Conversational Agent (Chat Layer)

The main interaction layer is an LLM-based agent responsible for generating responses.

- Model: `openai/gpt-oss-safeguard-20b` (Via Groq API)
- Supports:
  - tool-based reasoning
  - contextual responses
  - tone adaptation

The agent:
- receives structured inputs (state, sentiment, risk)
- relies on tools for all financial computations
- avoids performing calculations internally

A short-term memory mechanism (checkpointing) is used to:
- maintain conversational continuity
- preserve recent interaction context

---

### 4 Prompt Engineering System

Prompts are dynamically refined using a separate LLM-based prompt rewriting step.

- Model: Changed at runtime, same model the prompt is being re-written for
- Input:
  - original prompt
  - optional feedback
- Output:
  - improved system prompt

This approach:
- improves instruction clarity
- aligns prompts with model-specific behavior
- avoids manual prompt tuning

The system enforces strict constraints:
- output must be a prompt (not a response)
- no explanations or conversational text

---

### 5 Financial Computation Layer

All financial logic is implemented as deterministic functions and exposed as tools.

Includes:
- EMI calculation
- interest computation (monthly compounding)
- principal updates based on payments
- minimum payment logic
- projection of principal under non-payment

These tools are:
- directly used in backend logic
- invoked by the LLM for grounded responses

This ensures:
- consistency between system state and AI output
- elimination of hallucinated financial values

---

### 6 Risk Evaluation Module

Risk is computed based on deviation from an expected repayment trajectory.

Steps:
1. Compute expected principal using amortization logic  
2. Track actual principal from user behavior  
3. Calculate deviation:

   (actual - expected) / initial_principal

4. Clamp deviation:
   max(0, deviation)

5. Classify risk:
   - LOW
   - MEDIUM (≥ 5%)
   - HIGH (≥ 7%)

This module enables:
- behavior-aware classification
- dynamic feedback to the user
- integration with tone control in the conversational agent

## How to Run

```bash
# Create virtual environment
py -3.12 -m venv my_venv

# Activate
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file using sample
# Refer: app/agent/sample.env → copy contents and create .env in root

# Run application
py -m streamlit run app/streamlit/poc_chat.py
```


